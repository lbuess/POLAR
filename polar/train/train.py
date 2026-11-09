import os
import math
import time
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Custom imports (package-qualified so module execution is unambiguous)
from polar.modernbert import ModernBertForPolarRegression
from polar.train.dataset import PolarDataset


def evaluate(model, dataloader, device, error_cols):
    model.eval()
    total_loss = 0
    head_losses = {col: 0.0 for col in error_cols}
    num_batches = len(dataloader)

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Compute total loss
            loss = outputs.loss
            total_loss += float(loss.float().item())
            preds = outputs.all_logits

            # Compute head losses
            for i, col in enumerate(error_cols):
                head_pred = preds[i].squeeze(-1)
                head_label = labels[:, i]
                head_losses[col] += criterion(head_pred, head_label).item()

    avg_total_loss = total_loss / num_batches
    metrics = {"val/loss": avg_total_loss}

    for col, h_loss in head_losses.items():
        metrics[f"val/{col}_loss"] = h_loss / num_batches

    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="polar")
def main(cfg: DictConfig):
    print("Config: \n", OmegaConf.to_yaml(cfg))

    # Run toggles: train and eval are independent booleans.
    run_cfg = cfg.get("run", {})
    do_train = run_cfg.get("train", False)
    do_eval = run_cfg.get("eval", False)
    if not do_train and not do_eval:
        raise ValueError("Invalid run config: both run.train and run.eval are false.")

    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Set device (GPU required for both train and eval)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for both training and evaluation, but no CUDA device was detected."
        )
    device = torch.device("cuda" if cfg.training.device == "auto" else cfg.training.device)
    if device.type != "cuda":
        raise ValueError("Set training.device to 'cuda' or 'auto'.")
    print(f"Using device: {device}")

    # -----------------------
    # Load model/tokenizer
    # -----------------------
    model = None
    tokenizer = None

    if do_train:
        # Load tokenizer
        tokenizer_kwargs = {"trust_remote_code": True}
        if cfg.model.get("padding_side", None) is not None:
            tokenizer_kwargs["padding_side"] = cfg.model.padding_side
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, **tokenizer_kwargs)
        print("Tokenizer initialized.")

        # Load model
        config = AutoConfig.from_pretrained(cfg.model.name, trust_remote_code=True)
        config.trust_remote_code = bool(cfg.model.get("trust_remote_code", True))
        config.num_error_categories = len(cfg.data.error_cols)
        config.use_presence_loss = bool(cfg.model.get("use_presence_loss", False))
        config.presence_loss_weight = float(cfg.model.get("presence_loss_weight", 1.0))
        config.presence_label_threshold = float(
            cfg.model.get("presence_label_threshold", 0.0)
        )
        config.name = cfg.model.name
        model = ModernBertForPolarRegression.from_base_model(
            config, attn_implementation="flash_attention_2"
        )
        model.to(device)
        print("Model initialized.")
    else:
        # Eval-only mode: load checkpoint
        pass # TODO: implement eval-only mode

    # Training
    if do_train:
        # -----------------------
        # Load Datasets
        # ----------------------
        train_files = [
            hydra.utils.to_absolute_path(str(path)) for path in cfg.data.train_files
        ]
        val_files = [hydra.utils.to_absolute_path(str(path)) for path in cfg.data.val_files]

        # Train Dataset + Dataloader
        train_dataset = PolarDataset(
            csv_files=train_files,
            tokenizer=tokenizer,
            ref_col=cfg.data.ref_col,
            cand_col=cfg.data.cand_col,
            error_cols=cfg.data.error_cols,
            region_col=cfg.data.get("region_col", None),
            max_length=cfg.data.max_length,
        )
        if bool(cfg.training.get("debug", False)):
            debug_n = int(cfg.training.get("debug_num_samples", 100))
            train_dataset = Subset(train_dataset, range(min(debug_n, len(train_dataset))))
            print(f"Debug mode: using {len(train_dataset)} train samples")
        train_dataloader = DataLoader(
            train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4
        )
        print(f"Loaded {len(train_dataset)} train samples")

        # Val Dataset + Dataloader
        val_dataset = PolarDataset(
            csv_files=val_files,
            tokenizer=tokenizer,
            ref_col=cfg.data.ref_col,
            cand_col=cfg.data.cand_col,
            error_cols=cfg.data.error_cols,
            region_col=cfg.data.get("region_col", None),
            max_length=cfg.data.max_length,
        )
        if bool(cfg.training.get("debug", False)):
            debug_n = int(cfg.training.get("debug_num_samples", 100))
            val_dataset = Subset(val_dataset, range(min(debug_n, len(val_dataset))))
            print(f"Debug mode: using {len(val_dataset)} val samples")
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4
        )
        print(f"Loaded {len(val_dataset)} val samples")

        # -----------------------
        # Setup hyperparameters
        # ----------------------

        # Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
        gradient_accumulation_steps = int(
            cfg.training.get("gradient_accumulation_steps", 1)
        )
        # Scheduler steps should correspond to optimizer updates (not micro-batches).
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * int(cfg.training.epochs)
        warmup_steps_cfg = cfg.training.get("warmup_steps", None)
        warmup_ratio_cfg = cfg.training.get("warmup_ratio", None)
        if warmup_steps_cfg is not None:
            num_warmup_steps = int(warmup_steps_cfg)
        elif warmup_ratio_cfg is not None:
            warmup_ratio = float(warmup_ratio_cfg)
            if not 0.0 <= warmup_ratio <= 1.0:
                raise ValueError(
                    f"Invalid training.warmup_ratio={warmup_ratio}. Expected a value in [0.0, 1.0]."
                )
            num_warmup_steps = int(num_training_steps * warmup_ratio)
        else:
            num_warmup_steps = 0

        scheduler_name = str(cfg.training.get("scheduler", "linear")).lower()
        if scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_name == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(
                f"Invalid training.scheduler='{scheduler_name}'. Expected one of: linear, cosine."
            )

        # Output directory
        output_dir = hydra.utils.to_absolute_path(cfg.training.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to: {output_dir}")

        # -----------------------
        # Training Loop
        # ----------------------
        train_start_time = time.perf_counter()
        update_step = 0
        best_val_loss = float("inf")
        print(f"Training for {cfg.training.epochs} epochs")
        for epoch in range(cfg.training.epochs):
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
            model.train()
            progress_bar = tqdm(train_dataloader, desc="Training")
            optimizer.zero_grad()
            total_loss = 0.0
            accum_counter = 0
            accum_loss_sum = 0.0
            accum_loss_count = 0
            for step, batch in enumerate(progress_bar):
                # Get batch data
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # If loss is NaN, skip this batch
                if torch.isnan(loss):
                    print(f"Loss is NaN at epoch {epoch}, batch {step}!")
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    accum_loss_sum = 0.0
                    accum_loss_count = 0
                    continue

                # Scale loss for gradient accumulation so the effective gradient matches a single large batch.
                (loss / gradient_accumulation_steps).backward()
                accum_counter += 1
                current_loss = float(loss.item())
                accum_loss_sum += current_loss
                accum_loss_count += 1

                # If last batch or gradient accumulation steps reached, update model and scheduler
                is_last_batch = step == (len(train_dataloader) - 1)
                if accum_counter == gradient_accumulation_steps or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # Log once per optimizer update (average over micro-batches in this update).
                    avg_update_loss = (
                        accum_loss_sum / max(1, accum_loss_count)
                        if accum_loss_count > 0
                        else float("nan")
                    )
                    metrics_to_log = {
                        "train/loss": float(avg_update_loss),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch + (step + 1) / len(train_dataloader),
                    }
                    wandb.log(metrics_to_log, step=update_step)
                    update_step += 1
                    accum_counter = 0
                    accum_loss_sum = 0.0
                    accum_loss_count = 0

                total_loss += current_loss
                progress_bar.set_postfix({"loss": current_loss})

            avg_train_loss = total_loss / max(1, len(train_dataloader))
            print(f"Average Train Loss: {avg_train_loss:.4f}")

            # -----------------------
            # Validation
            # ----------------------
            metrics = evaluate(model, val_dataloader, device, cfg.data.error_cols)
            val_loss = metrics["val/loss"]
            print(f"Validation Loss: {val_loss:.4f}")

            # Log all metrics
            wandb_log_dict = metrics.copy()
            wandb_log_dict.update(
                {
                    "train/epoch_loss": avg_train_loss,
                    "epoch": epoch + 1,
                }
            )
            wandb.log(wandb_log_dict)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best validation loss! Saving model...")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

        train_seconds = time.perf_counter() - train_start_time
        print(
            f"Total training time: {train_seconds:.1f}s ({train_seconds/60.0:.1f} min)"
        )

    # -----------------------
    # Evaluation (eval-only or after training)
    # -----------------------
    if do_eval:
        pass # TODO: implement eval-only mode

    wandb.finish()


if __name__ == "__main__":
    main()
