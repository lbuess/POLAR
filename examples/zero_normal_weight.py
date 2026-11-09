"""Score candidates with the matched-normal-findings weight set to 0.

POLAR credits matched normal findings with weight w_n (normal_match_weight,
default 0.5). Reports in datasets such as CT-RATE contain a lot of normal
template text ("Trachea, both main bronchi are open. ... Thoracic aorta
diameter is normal. ..."). A candidate can reproduce all of it and still miss
every abnormal finding, and the matched normals then pull its score up.
Setting normal_match_weight=0.0 removes that credit, so the score depends only
on matched abnormal findings and errors. For reports with a lot of normal
template text this can be the more useful setting.

This example compares two chest CT candidates against the same reference:
  Candidate 1 copies all normal template sentences but misses both abnormal
              findings (pleural effusion, kidney calculi).
  Candidate 2 is short but keeps the pleural effusion.
With normal_match_weight=0.0, candidate 2 scores higher than candidate 1.

See the appendix of RadAgent: A tool-using AI agent for stepwise
interpretation of chest computed tomography (https://arxiv.org/abs/2604.15231)
for this setting in use.

From the repository root:
    python examples/zero_normal_weight.py
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch

from polar import compute_batch, load_model_and_tokenizer

# --- edit these ------------------------------------------------------------
CHECKPOINT = "lbuess/POLAR"  # released weights from the Hub, or a local checkpoint dir
REGION = "full report"

REFERENCE = (
    "Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, "
    "size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not "
    "observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening "
    "was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral "
    "hilar-axillary pathological dimensions were detected. Significant bilateral pleural effusion. "
    "Millimetric calculi images were observed in both kidneys."
)

CANDIDATES = {
    "Candidate 1": (
        "Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, "
        "size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not "
        "observed. Thoracic esophagus calibration was normal and no significant tumoral wall "
        "thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, "
        "subcarinal or bilateral hilar-axillary pathological dimensions were detected. No pleural "
        "effusion observed."
    ),
    "Candidate 2": (
        "Mediastinal main vascular structures, heart contour, size are normal. "
        "Significant bilateral pleural effusion."
    ),
}

# Scoring weight for matched normal findings (w_n). The default is 0.5;
# 0.0 means reproduced normal findings add nothing to the score.
NORMAL_MATCH_WEIGHT = 0.0
# ---------------------------------------------------------------------------

LABELS = "abcdefgh"
wrap = lambda text: textwrap.fill(text, width=100, subsequent_indent=" " * 14)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(CHECKPOINT)
model.to(device).eval()

names = list(CANDIDATES)
results = compute_batch(
    model=model,
    tokenizer=tokenizer,
    reference_reports=[REFERENCE] * len(names),
    candidate_reports=[CANDIDATES[name] for name in names],
    regions=[REGION] * len(names),
    device=device,
    normal_match_weight=NORMAL_MATCH_WEIGHT,
)

print("=" * 100)
print(f"Reference  (region: {REGION}, normal_match_weight={NORMAL_MATCH_WEIGHT})")
print("=" * 100)
print(f"  reference : {wrap(REFERENCE)}")

for name, result in zip(names, results):
    errors = "  ".join(f"{k}={v}" for k, v in zip(LABELS, result["error_counts"]))
    print(f"\n{name}")
    print(f"  candidate : {wrap(CANDIDATES[name])}")
    print(f"  errors    : {errors}")
    print(f"  matched   : normal={result['matched_findings_normal']}  "
          f"abnormal={result['matched_findings_abnormal']}")
    print(f"  score     : {result['scores']:+.4f}")
