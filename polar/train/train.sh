#!/bin/bash
# Train the final POLAR model (Bio-ModernBERT-large, 2048-token context).
# Run from the repository root. Any Hydra override can be appended, e.g.:
#   bash polar/train/train.sh training.batch_size=32 wandb.mode=online
set -euo pipefail

python -m polar.train.train --config-name=polar "$@"
