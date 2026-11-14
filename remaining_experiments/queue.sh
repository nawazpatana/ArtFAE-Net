#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ArtFAE-Net
while ! /workspace/.venv/bin/python -c 'import json; assert json.load(open("remaining_experiments/faithfulness/run.json"))["status"]=="complete"' 2>/dev/null; do sleep 15; done
HF_HUB_OFFLINE=1 /workspace/.venv/bin/python remaining_experiments/permutation.py > remaining_experiments/permutation.log 2>&1
HF_HUB_OFFLINE=1 /workspace/.venv/bin/python remaining_experiments/resources.py R1 > remaining_experiments/resources_R1.log 2>&1
HF_HUB_OFFLINE=1 /workspace/.venv/bin/python remaining_experiments/resources.py PROBE > remaining_experiments/resources_PROBE.log 2>&1
/workspace/.venv/bin/python remaining_experiments/analyse.py > remaining_experiments/analysis.log 2>&1
