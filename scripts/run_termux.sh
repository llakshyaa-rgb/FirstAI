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
if [ ! -f checkpoints/firstgpt.best.pt ] && [ -f /sdcard/Download/firstgpt.best.pt ]; then
  cp /sdcard/Download/firstgpt.best.pt checkpoints/firstgpt.best.pt
  echo "Imported checkpoint from Android Downloads."
fi
if [ ! -f checkpoints/firstgpt.best.pt ] && [ ! -f checkpoints/orbit-instruct.pt ]; then
  echo "No trained checkpoint found."
  echo "Download firstgpt.best.pt from Colab to Android Downloads, then run this command again."
  exit 2
fi
CHECKPOINT=checkpoints/orbit-instruct.pt
[ -f "$CHECKPOINT" ] || CHECKPOINT=checkpoints/firstgpt.best.pt
export ORBIT_WORKSPACE_ROOT=/root/FirstAI/workspace
export ORBIT_PRIVATE_MEDIA_ONLY=true
export ORBIT_CPU_THREADS=4
export ORBIT_TASK_WORKERS=2
export ORBIT_AGENT_WORKERS=2
export ORBIT_GPU_MEMORY_FRACTION=0
echo "Starting Orbit on CPU with checkpoint: $CHECKPOINT"
echo "Open http://127.0.0.1:8767 on this Android phone."
exec .venv/bin/python -m orbit.server --checkpoint "$CHECKPOINT" --state orbit_state.sqlite3 --host 127.0.0.1 --port 8767
'
