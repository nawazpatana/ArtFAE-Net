#!/usr/bin/env bash
# Local pilot of the ArtFAE-Net study on this server, in a tmux session (survives VS Code disconnecting).
# STUDY_MODE=pilot (banner on every page), SQLite storage in study/private/pilot_real/. Nothing goes to Hugging Face:
# HF variables are cleared and storage is forced to sqlite. Listens on 127.0.0.1 only; open it via VS Code port forwarding.
#
#   study/run_local_pilot.sh start     start the server in tmux session "artfae-pilot"
#   study/run_local_pilot.sh status    is it running? (health check)
#   study/run_local_pilot.sh attach    watch the server log (detach again with Ctrl-b d)
#   study/run_local_pilot.sh stop      stop the server
#   study/run_local_pilot.sh export    write trials.csv and participants.csv to study/private/pilot_real/export/
set -euo pipefail
umask 077  # pilot data, logs and exports: owner only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION=artfae-pilot
PORT=8602
DATA="$ROOT/study/private/pilot_real"
DB="$DATA/pilot.sqlite"
PY="$ROOT/artfae_demo/.venv/bin/python"
STREAMLIT="$ROOT/artfae_demo/.venv/bin/streamlit"
TMPD=/workspace/ui_tools/tmp  # the root filesystem (/tmp) is full

case "${1:-}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running (tmux session $SESSION)"; exit 0; fi
    mkdir -p "$DATA" "$TMPD"
    chmod 700 "$DATA"; find "$DATA" -maxdepth 1 -type f -exec chmod 600 {} +
    "$PY" -I -B "$ROOT/study/frozen.py" >/dev/null || { echo "frozen stimulus set does not verify; not starting"; exit 1; }
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
      "env -u HF_TOKEN -u DATASET_REPO -u SPACE_ID -u CONSENT_APPROVED -u STUDY_DEV \
           STUDY_MODE=pilot STUDY_STORAGE=sqlite STUDY_DB='$DB' TMPDIR='$TMPD' PYTHONDONTWRITEBYTECODE=1 \
       '$STREAMLIT' run study/study_app.py --server.address 127.0.0.1 --server.port $PORT \
           --server.headless true --browser.gatherUsageStats false 2>&1 | tee -a '$DATA/server.log'"
    for _ in $(seq 1 60); do
      curl -sf "http://127.0.0.1:$PORT/_stcore/health" >/dev/null && { echo "running: http://localhost:$PORT  (data: $DB)"; exit 0; }
      sleep 1
    done
    echo "did not become healthy; see $DATA/server.log"; exit 1 ;;
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "not running" ;;
  status)
    if curl -sf "http://127.0.0.1:$PORT/_stcore/health" >/dev/null; then echo "running on port $PORT; data: $DB"; else echo "not running"; fi ;;
  attach)
    tmux attach -t "$SESSION" ;;
  export)
    "$PY" -I -B "$ROOT/study/export.py" "$DATA/export" --sqlite "$DB" ;;
  *)
    sed -n '2,13p' "$0"; exit 2 ;;
esac
