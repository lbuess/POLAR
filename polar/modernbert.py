from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

import torch
import torch.nn as nn


@dataclass
class PolarRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    all_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_presence_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ModernBertForPolarRegression(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, attn_implementation: str = None):
        super().__init__(config)

        # ModernBERT architecture only; weights come from a POLAR checkpoint
        # (from_pretrained) or from the base model (from_base_model, for training).
        # Inference defaults to SDPA attention (fp32-safe); training passes
        # flash_attention_2 explicitly.
        self.encoder = AutoModel.from_config(
            config, attn_implementation=attn_implementation or "sdpa"
        )
        self.dropout = nn.Dropout(0.1)  # Dropout rate

        # Number of heads (error categories + matched findings)
        self.num_heads = getattr(config, "num_error_categories", 0)
        if self.num_heads == 0:
            raise ValueError("num_error_categories must be set in the config")

        # Regressors for each head
        self.head_regressors = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(self.num_heads)]
        )

        # Optional: add a parallel "presence" head per regressor (binary: target > threshold),
        # as in CREPE's auxiliary objective (extra presence loss per category).
        self.use_presence_loss = bool(getattr(config, "use_presence_loss", False))
        self.presence_loss_weight = float(getattr(config, "presence_loss_weight", 1.0))
        self.presence_label_threshold = float(
            getattr(config, "presence_label_threshold", 0.0)
        )
        self.head_presence = (
            nn.ModuleList(
                [nn.Linear(config.hidden_size, 1) for _ in range(self.num_heads)]
            )
            if self.use_presence_loss
            else None
        )

        # Required by transformers 5.x (sets up weight-tying metadata).
        self.post_init()
        self._init_regressor_weights()

    @classmethod
    def from_base_model(cls, config, attn_implementation: str = None):
        """Build a model for training: pretrained encoder (config.name), fresh heads."""
        model = cls(config, attn_implementation=attn_implementation)
        model.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=config.name,
            config=config,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        return model

    def _init_regressor_weights(self):
        def init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

        for regressor in self.head_regressors:
            init_weights(regressor)
        if self.head_presence is not None:
            for head in self.head_presence:
                init_weights(head)

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        inputs_embeds = kwargs.get("inputs_embeds")
        labels = kwargs.get("labels")
        output_attentions = kwargs.get("output_attentions")
        output_hidden_states = kwargs.get("output_hidden_states")

        # Forward pass
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Get cls token
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        cls_output = self.dropout(cls_output)

        # Get regressor logits
        regressor_logits = tuple(reg(cls_output) for reg in self.head_regressors)

        # Get presence logits
        presence_logits = (
            tuple(head(cls_output) for head in self.head_presence)
            if self.head_presence is not None
            else None
        )

        # Compute loss
        loss = None
        if labels is not None:
            # Mean MSE across heads (POLAR trains multi-head regression).
            reg_loss_fct = nn.MSELoss()
            reg_total = sum(
                reg_loss_fct(regressor_logits[i].squeeze(-1), labels[:, i])
                for i in range(self.num_heads)
            )
            reg_loss = reg_total / self.num_heads
            loss = reg_loss

            # Optional auxiliary "presence" loss: BCE on whether each head's target is > threshold.
            if presence_logits is not None:
                presence_targets = (labels > self.presence_label_threshold).float()
                presence_loss_fct = nn.BCEWithLogitsLoss()
                presence_total = sum(
                    presence_loss_fct(
                        presence_logits[i].squeeze(-1), presence_targets[:, i]
                    )
                    for i in range(self.num_heads)
                )
                presence_loss = presence_total / self.num_heads
                loss = loss + (self.presence_loss_weight * presence_loss)

        return PolarRegressionOutput(
            loss=loss,
            all_logits=regressor_logits,
            all_presence_logits=presence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_model_and_tokenizer(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except ValueError as exc:
        # Some saved checkpoints carry tokenizer_class=TokenizersBackend, which older
        # transformers versions cannot import from local files. Fall back to base model.
        base_model_name = getattr(config, "name", None)
        if not base_model_name:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
    model = ModernBertForPolarRegression.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
    )
    return model, tokenizer


def compute(
    model,
    tokenizer,
    reference_report: str,
    candidate_report: str,
    region: str = "full report",
    device: str = "cpu",
    max_length: int = 2048,
    abnormal_match_weight: float = 2.0,
    normal_match_weight: float = 0.5,
    error_ab_weight: float = 2.0,
    error_fn_weight: float = 2.0,
    error_other_weight: float = 1.0,
) -> Dict[str, object]:
    condition_text = f"Condition: {region}"
    reference_text = f"Reference: {reference_report}"
    candidate_text = f"Candidate: {candidate_report}"

    if hasattr(tokenizer, "sep_token") and tokenizer.sep_token is not None:
        sep = f" {tokenizer.sep_token} "
        joint_text = sep.join([condition_text, reference_text, candidate_text])
    else:
        joint_text = f"{condition_text}\n{reference_text}\n{candidate_text}\n"

    inputs = tokenizer(
        joint_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        predicted_error_counts_raw = [
            float(logit.item()) for logit in outputs.all_logits
        ]

    predicted_error_counts = [
        int(math.floor(max(0.0, count) + 0.5))
        for count in predicted_error_counts_raw[:8]
    ]
    matched_findings_normal = int(
        math.floor(max(0.0, predicted_error_counts_raw[8]) + 0.5)
    )
    matched_findings_abnormal = int(
        math.floor(max(0.0, predicted_error_counts_raw[9]) + 0.5)
    )
    matched_findings = matched_findings_normal + matched_findings_abnormal

    if (
        abnormal_match_weight < 0.0
        or normal_match_weight < 0.0
        or error_ab_weight < 0.0
        or error_fn_weight < 0.0
        or error_other_weight < 0.0
    ):
        raise ValueError("All score weights must be >= 0.0")

    sig_fp   = predicted_error_counts[0]
    sig_fn   = predicted_error_counts[1]
    sig_attr = sum(predicted_error_counts[2:])

    # W_ref: total abnormal reference weight — anchors the score scale.
    # Includes FN so that W_ref = (TP_ab + FN) * w_ab regardless of candidate quality.
    W_ref = (matched_findings_abnormal + sig_fn) * abnormal_match_weight
    W_n   = matched_findings_normal * normal_match_weight

    C = matched_findings_abnormal * abnormal_match_weight + matched_findings_normal * normal_match_weight

    # FN is now an explicit penalty alongside FP and attribute errors.
    E = sig_fp * error_ab_weight + sig_fn * error_fn_weight + sig_attr * error_other_weight

    denom = W_ref + W_n + E
    scores = (C - E) / denom if denom > 0.0 else 1.0

    return {
        "scores": scores,
        "error_counts": predicted_error_counts,
        "matched_findings": matched_findings,
        "matched_findings_normal": matched_findings_normal,
        "matched_findings_abnormal": matched_findings_abnormal,
        "weighted_matched": C,
        "weighted_errors": E,
    }


def compute_batch(
    model,
    tokenizer,
    reference_reports: list[str],
    candidate_reports: list[str],
    regions: Optional[list[str]] = None,
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 2048,
    abnormal_match_weight: float = 2.0,
    normal_match_weight: float = 0.5,
    error_ab_weight: float = 2.0,
    error_fn_weight: float = 2.0,
    error_other_weight: float = 1.0,
) -> list[Dict[str, object]]:
    """Score several report pairs; returns one compute()-style dict per pair.

    Mirrors compute() step by step (same input text, padding to max_length,
    rounding and scoring); only the forward pass runs batch_size pairs at once.
    """
    if len(reference_reports) != len(candidate_reports):
        raise ValueError("reference_reports and candidate_reports must have the same length")
    if regions is None:
        regions = ["full report"] * len(reference_reports)
    if len(regions) != len(reference_reports):
        raise ValueError("regions must have the same length as reference_reports")
    if (
        abnormal_match_weight < 0.0
        or normal_match_weight < 0.0
        or error_ab_weight < 0.0
        or error_fn_weight < 0.0
        or error_other_weight < 0.0
    ):
        raise ValueError("All score weights must be >= 0.0")

    # Same input text as compute()
    if hasattr(tokenizer, "sep_token") and tokenizer.sep_token is not None:
        sep = f" {tokenizer.sep_token} "
    else:
        sep = None
    joint_texts = []
    for reference_report, candidate_report, region in zip(
        reference_reports, candidate_reports, regions
    ):
        condition_text = f"Condition: {region}"
        reference_text = f"Reference: {reference_report}"
        candidate_text = f"Candidate: {candidate_report}"
        if sep is not None:
            joint_texts.append(sep.join([condition_text, reference_text, candidate_text]))
        else:
            joint_texts.append(f"{condition_text}\n{reference_text}\n{candidate_text}\n")

    results = []
    for start in range(0, len(joint_texts), batch_size):
        inputs = tokenizer(
            joint_texts[start:start + batch_size],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        # all_logits: one (batch, 1) tensor per head -> (batch, num_heads)
        raw_batch = torch.cat([logit.float() for logit in outputs.all_logits], dim=-1).cpu().tolist()

        # Same rounding and scoring as compute()
        for predicted_error_counts_raw in raw_batch:
            predicted_error_counts = [
                int(math.floor(max(0.0, count) + 0.5))
                for count in predicted_error_counts_raw[:8]
            ]
            matched_findings_normal = int(
                math.floor(max(0.0, predicted_error_counts_raw[8]) + 0.5)
            )
            matched_findings_abnormal = int(
                math.floor(max(0.0, predicted_error_counts_raw[9]) + 0.5)
            )
            matched_findings = matched_findings_normal + matched_findings_abnormal

            sig_fp   = predicted_error_counts[0]
            sig_fn   = predicted_error_counts[1]
            sig_attr = sum(predicted_error_counts[2:])

            W_ref = (matched_findings_abnormal + sig_fn) * abnormal_match_weight
            W_n   = matched_findings_normal * normal_match_weight
            C = matched_findings_abnormal * abnormal_match_weight + matched_findings_normal * normal_match_weight
            E = sig_fp * error_ab_weight + sig_fn * error_fn_weight + sig_attr * error_other_weight

            denom = W_ref + W_n + E
            scores = (C - E) / denom if denom > 0.0 else 1.0

            results.append({
                "scores": scores,
                "error_counts": predicted_error_counts,
                "matched_findings": matched_findings,
                "matched_findings_normal": matched_findings_normal,
                "matched_findings_abnormal": matched_findings_abnormal,
                "weighted_matched": C,
                "weighted_errors": E,
            })
    return results
