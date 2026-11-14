#!/usr/bin/env bash
# Five-seed ladder (val only): R1 | R2b | R3 (lambda 0.3) | R3g | R4b. Existing run folders are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=/workspace/.venv/bin/python
declare -A FLAGS=(
    [R1]=""
    [R2b_sampler]="--sampler"
    [R3_lam0.3]="--sampler --lambda-present 0.3"
    [R3g_gate]="--sampler --lambda-present 0.3 --regional-gate -2.0"
    [R4b_textq_gate]="--sampler --lambda-present 0.3 --text-queries --regional-gate -2.0"
)
mkdir -p logs/train
for seed in 42 43 44 45 46; do
    for run in R1 R2b_sampler R3_lam0.3 R3g_gate R4b_textq_gate; do
        out="runs/${run}_s${seed}"
        if [ -f "$out/val_logits.npy" ]; then echo "$(date +%T) skip $out (exists)"; continue; fi
        echo "$(date +%T) train $out ${FLAGS[$run]}"
        $PY -m artfae_balance.train --out "$out" --seed "$seed" ${FLAGS[$run]} > "logs/train/${run}_s${seed}.log" 2>&1
        $PY -m artfae_balance.evaluate --manifest data/artemis_balanced/val.csv \
            --logits "$out/val_logits.npy" --out "$out/val_report.json" > /dev/null
        echo "$(date +%T) done  $out: $(tail -1 "logs/train/${run}_s${seed}.log")"
    done
done
echo "$(date +%T) LADDER COMPLETE"
