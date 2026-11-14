#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Independent Android installation. Does not require Colab or a transferred checkpoint.
if ! command -v pkg >/dev/null 2>&1; then
  echo "Run this in Termux."
  exit 1
fi

echo "[1/5] Installing Termux prerequisites..."
pkg update -y
pkg install -y proot-distro git termux-api
if ! proot-distro login debian -- true >/dev/null 2>&1; then
  echo "[2/5] Installing Debian..."
  proot-distro install debian
else
  echo "[2/5] Debian already installed."
fi

echo "[3/5] Installing isolated Orbit environment..."
proot-distro login debian -- bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip git ca-certificates curl ffmpeg libopenblas0 libgomp1 nodejs npm
if [ ! -d /root/FirstAI/.git ]; then
  git clone https://github.com/llakshyaa-rgb/FirstAI.git /root/FirstAI
else
  git -C /root/FirstAI pull --ff-only
fi
cd /root/FirstAI
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-design.txt
mkdir -p checkpoints workspace data/local
'

echo "[4/5] Enabling Android shared storage..."
termux-setup-storage || true

echo "[5/5] Probing Android GPU access (diagnostic only)..."
if command -v termux-info >/dev/null 2>&1; then termux-info | head -n 35 || true; fi
echo
echo "Independent Orbit environment installed at Debian:/root/FirstAI"
echo "It has its own memory database, settings, checkpoints and workspace."
echo "No Colab connection or Colab checkpoint is required to start the local app."
echo "If no checkpoint exists, Orbit starts in tool-only mode and reports model-not-ready."
echo "To train a small local FirstGPT checkpoint, run:"
echo "  proot-distro login debian -- bash -lc 'cd /root/FirstAI && .venv/bin/python train.py --data data/input.txt --out checkpoints/firstgpt.pt --steps 1000 --batch-size 2 --block-size 256 --layers 2 --heads 2 --width 64'"
echo "For Android GPU diagnostics, run:"
echo "  bash scripts/probe_android_gpu.sh"
echo "Then launch with: bash scripts/run_termux.sh"
