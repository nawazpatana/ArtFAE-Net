#!/usr/bin/env bash
# L/14@336 (7x7-pooled) R1 and R3 (lambda 0.3), seeds 42-44; same config, dims from the features. Val only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
PY=/workspace/.venv/bin/python
until grep -q 'all splits done' logs/extract_l14_test.log 2>/dev/null; do
    if grep -qs 'exit_code=[1-9]\|Traceback' logs/extract_l14_*.log; then echo "extraction failed"; exit 1; fi
    sleep 10
done
grep -H exit_code logs/extract_l14_*.log
for seed in 42 43 44; do
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
