#!/usr/bin/env bash
# Revision Phase A: regenerate every table in runs/revision_A from existing outputs (no training, no test re-inference).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
PY=/workspace/.venv/bin/python
for s in t1_baselines t2_ties t3_confusion t4_seen_unseen t5_params t6_subset_stats t7_feature_stats t8_tost; do
  echo "== $s"; $PY $s.py > ../logs/$s.log 2>&1
done
$PY verify_untouched.py | tee ../logs/verify_untouched.log
$PY build_readme.py
