#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
if ! command -v proot-distro >/dev/null 2>&1; then
  echo "Run bash scripts/setup_termux.sh first."
  exit 1
fi
proot-distro login debian -- bash -lc '
set -euo pipefail
cd /root/FirstAI
mkdir -p checkpoints workspace
CHECKPOINT=checkpoints/orbit-instruct.pt
[ -f "$CHECKPOINT" ] || CHECKPOINT=checkpoints/orbit-mobile-instruct.pt
[ -f "$CHECKPOINT" ] || CHECKPOINT=checkpoints/firstgpt.best.pt
[ -f "$CHECKPOINT" ] || CHECKPOINT=checkpoints/firstgpt.pt
export ORBIT_WORKSPACE_ROOT=/root/FirstAI/workspace
export ORBIT_PRIVATE_MEDIA_ONLY=true
export ORBIT_CPU_THREADS=4
export ORBIT_TASK_WORKERS=2
export ORBIT_AGENT_WORKERS=2
if [ -f "$CHECKPOINT" ]; then
  echo "Starting independent local Orbit with $CHECKPOINT"
else
  echo "Starting independent local Orbit in tool-only mode: no local checkpoint found."
  echo "Train locally or place a compatible checkpoint in /root/FirstAI/checkpoints."
fi
echo "Open http://127.0.0.1:8767 on this Android phone."
exec .venv/bin/python -m orbit.server --checkpoint "$CHECKPOINT" --state orbit_state.sqlite3 --host 127.0.0.1 --port 8767
'
