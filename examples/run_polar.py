"""Score report pairs with POLAR: one pair with compute(), several with compute_batch().

From the repository root:
    python examples/run_polar.py
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import torch

from polar import compute, compute_batch, load_model_and_tokenizer

# --- edit these ------------------------------------------------------------
CHECKPOINT = "lbuess/POLAR"  # released weights from the Hub, or a local checkpoint dir

REFERENCE = "No pleural effusion present."
CANDIDATE = "Pleural effusion present."
REGION = "full report"  # or an anatomy, e.g. "pleura"

# Several pairs for compute_batch()
PAIRS = [
    ("Left-sided pleural effusion. Normal heart size.", "Cardiomegaly. No pleural effusion noted.", "full report"),
    ("No pleural effusion present.", "Pleural effusion present.", "full report"),
    # Same pair, full report vs. conditioned on one anatomy: with "pleura" only
    # pleural findings are counted, so "Lungs are clear." drops out.
    ("Small right pleural effusion. Lungs are clear.", "Small right pleural effusion. Lungs are clear.", "full report"),
    ("Small right pleural effusion. Lungs are clear.", "Small right pleural effusion. Lungs are clear.", "pleura"),
    # Paraphrase from the RadEval divergence gallery: same finding, different
    # wording (no word overlap, so lexical metrics such as BLEU score it 0).
    ("Mild cardiomegaly.", "The cardiac silhouette is mildly enlarged.", "full report"),
]
BATCH_SIZE = 2
# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(CHECKPOINT)
model.to(device).eval()

# --- one pair: compute() ----------------------------------------------------
result = compute(
    model=model,
    tokenizer=tokenizer,
    reference_report=REFERENCE,
    candidate_report=CANDIDATE,
    region=REGION,
    device=device,
    max_length=2048,
    abnormal_match_weight=2.0,
    normal_match_weight=0.5,
    error_ab_weight=2.0,
    error_fn_weight=2.0,
    error_other_weight=1.0,
)

LABELS = "abcdefgh"


def print_result(title, reference, candidate, region, result, note=""):
    """One pair: texts on top, counts and score below, aligned."""
    wrap = lambda text: textwrap.fill(text, width=100, subsequent_indent=" " * 14)
    errors = "  ".join(f"{k}={v}" for k, v in zip(LABELS, result["error_counts"]))
    print(f"\n{title}  (region: {region})")
    print(f"  reference : {wrap(reference)}")
    print(f"  candidate : {wrap(candidate)}")
    print(f"  errors    : {errors}")
    print(f"  matched   : normal={result['matched_findings_normal']}  "
          f"abnormal={result['matched_findings_abnormal']}")
    print(f"  score     : {result['scores']:+.4f}{note}")


print("=" * 100)
print("One pair with compute()")
print("=" * 100)
print_result("Pair", REFERENCE, CANDIDATE, REGION, result)

# --- several pairs: compute_batch() -----------------------------------------
references, candidates, regions = (list(x) for x in zip(*PAIRS))
batch_results = compute_batch(
    model=model,
    tokenizer=tokenizer,
    reference_reports=references,
    candidate_reports=candidates,
    regions=regions,
    device=device,
    batch_size=BATCH_SIZE,
)

# compute_batch() gives the same results as calling compute() pair by pair
single_results = [
    compute(model, tokenizer, ref, cand, region=region, device=device)
    for ref, cand, region in PAIRS
]

print("\n" + "=" * 100)
print(f"{len(PAIRS)} pairs with compute_batch(batch_size={BATCH_SIZE})")
print("=" * 100)
for i, ((ref, cand, region), batched, single) in enumerate(zip(PAIRS, batch_results, single_results), 1):
    note = "   (same as compute())" if batched == single else "   (DIFFERENT from compute()!)"
    print_result(f"Pair {i}", ref, cand, region, batched, note)

print("\nAll results identical to compute():", batch_results == single_results)
