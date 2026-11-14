#!/usr/bin/env bash
# Frozen CLIP feature extraction for the balanced ArtEmis splits.
# One log per split, every line timestamped: logs/extract_<split>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p logs features
for split in train val test; do
    log="logs/extract_${split}.log"
    {
        echo "start $(date -Is) manifest=data/artemis_balanced/${split}.csv rows=$(( $(wc -l < data/artemis_balanced/${split}.csv) - 1 ))"
        /workspace/.venv/bin/python -m artfae_balance.extract_features \
            --manifest "data/artemis_balanced/${split}.csv" --out "features/${split}" --batch 128 2>&1
        echo "exit_code=$? end $(date -Is)"
    } | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done > "$log"
done
echo "all splits done $(date -Is)" >> logs/extract_test.log
