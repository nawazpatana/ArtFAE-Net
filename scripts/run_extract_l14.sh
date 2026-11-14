#!/usr/bin/env bash
# CLIP ViT-L/14@336 features (fp32, same load_image + HOOK), patch tokens average-pooled to 7x7.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for split in train val test; do
    log="logs/extract_l14_${split}.log"
    {
        echo "start $(date -Is) manifest=data/artemis_balanced/${split}.csv"
        /workspace/.venv/bin/python -m artfae_balance.extract_features_backbone \
            --manifest "data/artemis_balanced/${split}.csv" --out "features_l14/${split}" \
            --model openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1 \
            --batch 64 --pool-grid 7 2>&1 | grep --line-buffered -v -i warn
        echo "exit_code=${PIPESTATUS[0]} end $(date -Is)"
    } | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done > "$log"
done
echo "all splits done $(date -Is)" >> logs/extract_l14_test.log
