"""Download the public POLAR-Synth subsets from Hugging Face into the training layout.

Writes dataset/<split>/<subset>_<split>.csv, the paths the training config
(polar/train/conf/data/default.yaml) reads. The dataset is gated: accept the
terms on https://huggingface.co/datasets/lbuess/POLAR-Synth and log in with
`hf auth login` first. Run from the repository root:

    python dataset/download.py
    python dataset/download.py --splits train val test
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError

REPO_ID = "lbuess/POLAR-Synth"
SUBSETS = ("ct-rate", "mr-rate")
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=SUBSETS)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=SPLITS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="target folder (default: this dataset/ folder)")
    args = parser.parse_args()

    for subset in args.subsets:
        for split in args.splits:
            try:
                cached = hf_hub_download(args.repo_id, f"{subset}/{split}.csv", repo_type="dataset")
            except GatedRepoError:
                sys.exit(
                    f"{args.repo_id} is gated. Accept the terms on "
                    f"https://huggingface.co/datasets/{args.repo_id} and log in with "
                    "`hf auth login`, then run this script again."
                )
            target = args.out_dir / split / f"{subset}_{split}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            # copyfile, not copy: the HF cache files are read-only, and copying
            # their permissions would make a re-run fail.
            shutil.copyfile(cached, target)
            print(f"{subset}/{split}.csv -> {target}")


if __name__ == "__main__":
    main()
