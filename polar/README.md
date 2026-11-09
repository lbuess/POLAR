# POLAR: Model, Inference and Training

This folder contains the POLAR model ([modernbert.py](modernbert.py)) and its training code ([train/](train/)). For installation and a quick start, see the [main README](../README.md).

## Contents

- [Model](#model)
- [Inference](#inference)
- [Training](#training)

## Model

POLAR fine-tunes [Bio-ModernBERT-large](https://huggingface.co/thomas-sounack/Bio-ModernBERT-large) with 10 regression heads. The input is a single sequence:

```
Condition: <region> [SEP] Reference: <reference report> [SEP] Candidate: <candidate report>
```

It is truncated or padded to 2048 tokens. Each head reads the `[CLS]` embedding and predicts one count:

| Head | Prediction |
|---|---|
| `sig_a` … `sig_h` | Clinically significant errors in categories (a)–(h) |
| `matched_normal` | Normal findings correctly reproduced in the candidate |
| `matched_abnormal` | Abnormal findings correctly reproduced in the candidate |

<details>
<summary><b>Error categories</b></summary>

| Category | Description |
|---|---|
| (a) | False report of a finding in the candidate |
| (b) | Missing a finding present in the reference |
| (c) | Misidentification of anatomic location, laterality, or position |
| (d) | Misassessment of severity or extent |
| (e) | Mentioning a comparison not present in the reference |
| (f) | Omitting a comparison detailing a change from a prior study |
| (g) | Misstatement of quantitative details (size, number, measurements) |
| (h) | Misstatement of descriptive attributes (morphology, shape, margin, appearance) |

</details>

The predicted counts are rounded to integers and combined into the POLAR score:

```
W_ref  = (matched_abnormal + sig_b) × w_ab
W_n    = matched_normal × w_n
C      = matched_abnormal × w_ab + matched_normal × w_n
E      = sig_a × w_fp + sig_b × w_fn + (sig_c + … + sig_h) × w_other
score  = (C − E) / (W_ref + W_n + E)        # 1.0 if the denominator is 0
```

## Inference

```python
from polar import load_model_and_tokenizer, compute

model, tokenizer = load_model_and_tokenizer("lbuess/POLAR")  # or a local checkpoint dir
model.to("cuda").eval()

result = compute(
    model=model,
    tokenizer=tokenizer,
    reference_report="Left-sided pleural effusion. Normal heart size.",
    candidate_report="Cardiomegaly. No pleural effusion noted.",
    region="full report",
    device="cuda",
)
```

### Arguments of `compute()`

| Argument | Default | Description |
|---|---|---|
| `region` | `"full report"` | What to evaluate; see [Region](#region) |
| `device` | `"cpu"` | Device the model is on |
| `max_length` | `2048` | Context length; keep at the training value |
| `abnormal_match_weight` (`w_ab`) | `2.0` | Weight of matched abnormal findings |
| `normal_match_weight` (`w_n`) | `0.5` | Weight of matched normal findings |
| `error_ab_weight` (`w_fp`) | `2.0` | Penalty for false findings (a) |
| `error_fn_weight` (`w_fn`) | `2.0` | Penalty for missed findings (b) |
| `error_other_weight` (`w_other`) | `1.0` | Penalty for attribute errors (c)–(h) |

### Region

| `region` | Evaluates |
|---|---|
| `"full report"` | all findings in the report |
| an anatomy, e.g. `"lung"`, `"pleura"`, `"heart"`, `"liver and biliary tree"`, `"brain"` | only findings for that anatomy |

Use lowercase anatomy names, since anatomies appeared that way during training. The full list is in `ANATOMY_SYNONYMS` in [train/dataset_utils.py](train/dataset_utils.py).

### Output

| Key | Description |
|---|---|
| `scores` | POLAR score in [−1, 1] |
| `error_counts` | Predicted error counts for categories (a)–(h) |
| `matched_findings_normal` | Matched normal findings |
| `matched_findings_abnormal` | Matched abnormal findings |
| `matched_findings` | Sum of the two |
| `weighted_matched`, `weighted_errors` | `C` and `E` from the scoring formula |

### Several report pairs

`compute_batch()` scores lists of report pairs, running `batch_size` pairs per forward pass. It takes the same arguments as `compute()` (with `regions` as a list, `None` for full reports) and returns one result dict per pair, identical to calling `compute()` for each pair:

```python
from polar import compute_batch

results = compute_batch(
    model=model,
    tokenizer=tokenizer,
    reference_reports=["No pleural effusion present.", "Mild cardiomegaly."],
    candidate_reports=["Pleural effusion present.", "The cardiac silhouette is mildly enlarged."],
    regions=None,  # or one region per pair
    device="cuda",
    batch_size=16,
)
print([r["scores"] for r in results])
```

## Training

Training needs a CUDA GPU, the training dependencies and [FlashAttention 2](https://github.com/Dao-AILab/flash-attention). Install the versions the released model was trained with:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train]" "transformers==5.0.0"
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

This requires a GPU driver that supports CUDA 12.4.

Download the training data (`python dataset/download.py`, see [dataset/README.md](../dataset/README.md#download)), then run from the repository root:

```bash
bash polar/train/train.sh
```

This trains the released configuration ([train/conf/polar.yaml](train/conf/polar.yaml)):

| Setting | Value |
|---|---|
| Encoder | Bio-ModernBERT-large |
| Context length | 2048 tokens |
| Epochs | 10 |
| Effective batch size | 128 (64 × 2 gradient accumulation) |
| Optimizer | AdamW, lr 2e-5, linear schedule |
| Loss | per-head MSE + auxiliary presence loss |

Any setting can be overridden with [Hydra](https://hydra.cc/) syntax:

```bash
bash polar/train/train.sh training.batch_size=32 training.gradient_accumulation_steps=4
```

The checkpoint with the lowest validation loss is saved to `outputs/<model>_<data>/`, and `load_model_and_tokenizer` can load it.

Weights & Biases logging is off by default. To turn it on, add `wandb.mode=online wandb.entity=<your-entity>`.

### Files

```
modernbert.py          model, load_model_and_tokenizer, compute
train/
  train.py             Hydra training entry point
  dataset.py           input formatting and labels
  dataset_utils.py     anatomy synonyms for the conditioning text
  conf/
    polar.yaml         released configuration
    model/             encoder settings
    data/              data paths and label columns
  train.sh             training launcher
```
