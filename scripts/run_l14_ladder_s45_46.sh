#!/usr/bin/env bash
# L/14@336 (7x7-pooled) R1 and R3 (lambda 0.3), seeds 45-46 (completes 5 seeds); same config, dims from the features. Val only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
PY=/workspace/.venv/bin/python
for seed in 45 46; do
    for spec in "L14_R1|" "L14_R3_lam0.3|--sampler --lambda-present 0.3"; do
        name=${spec%%|*}; flags=${spec#*|}; out="runs/${name}_s${seed}"
        [ -f "$out/val_logits.npy" ] && { echo "skip $out"; continue; }
        echo "$(date +%T) train $out $flags"
        $PY -m artfae_balance.train_backbone --out "$out" --seed "$seed" \
            --train-features features_l14/train --val-features features_l14/val $flags > "logs/train/${name}_s${seed}.log" 2>&1
        $PY -m artfae_balance.evaluate --manifest data/artemis_balanced/val.csv \
            --logits "$out/val_logits.npy" --out "$out/val_report.json" > /dev/null
        echo "$(date +%T) done  $out: $(grep selected "logs/train/${name}_s${seed}.log")"
    done
done
echo "$(date +%T) L14 LADDER COMPLETE"
