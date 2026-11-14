#!/usr/bin/env bash
# Phase C: CLIP L/14@336 features for the frozen fresh holdout, with exactly the settings used for features_l14
# (scripts/run_extract_l14.sh): extract_features_backbone, same model + revision, batch 64, 7x7 pooled patches,
# whole-image attributes + describe() text. Run only after holdout.csv is frozen and hashed in the final selection.
# usage: bash extract_holdout_features.sh [PREP_DIR] [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/../../.."
PREP=${1:-runs/revision_B/fresh_holdout}
OUTD=${2:-features_l14_fresh_holdout}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
[ -e "$OUTD" ] && { echo "$OUTD exists; refusing to overwrite"; exit 1; }
/workspace/.venv/bin/python -m artfae_balance.extract_features_backbone \
    --manifest "$PREP/holdout.csv" --out "$OUTD" \
    --model openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 \
    --batch 64 --pool-grid 7 2>&1 | grep --line-buffered -v -i warn
sha256sum "$PREP/holdout.csv" "$OUTD"/*.npy
