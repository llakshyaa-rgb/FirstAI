#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

# Run from Termux (not inside Python):
#   bash scripts/setup_termux.sh
# This provisions Debian in proot so Linux ARM64 PyTorch wheels can be used.
# It does not modify the existing Termux Python installation.

if ! command -v pkg >/dev/null 2>&1; then
  echo "Run this script in the Termux app, not inside Debian or another shell."
  exit 1
fi

echo "[1/4] Installing Termux prerequisites..."
pkg update -y
pkg install -y proot-distro git

if proot-distro login debian -- true >/dev/null 2>&1; then
  echo "[2/4] Debian rootfs already exists."
else
  echo "[2/4] Installing Debian rootfs (this may take a while)..."
  proot-distro install debian
fi

echo "[3/4] Installing Orbit runtime inside Debian..."
proot-distro login debian -- bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip git ca-certificates curl   ffmpeg libopenblas0 libgomp1 nodejs npm
if [ ! -d /root/FirstAI/.git ]; then
  git clone https://github.com/llakshyaa-rgb/FirstAI.git /root/FirstAI
else
  git -C /root/FirstAI pull --ff-only
fi
cd /root/FirstAI
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
# CPU-only PyTorch: the Android phone has no CUDA device. Keep Colab for training.
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-design.txt
mkdir -p checkpoints workspace
'

echo "[4/4] Setting up shared storage access..."
if command -v termux-setup-storage >/dev/null 2>&1; then
  echo "Android may ask you to grant Termux storage permission."
  termux-setup-storage || true
fi

echo
echo "Termux + Debian runtime is provisioned."
echo "Next: download your trained checkpoint from Colab to Android Downloads."
echo "Expected file: /sdcard/Download/firstgpt.best.pt"
echo "Then start Orbit with:"
echo "  bash scripts/run_termux.sh"
echo
echo "Orbit will bind to 127.0.0.1:8767 only. Open http://127.0.0.1:8767 on this phone."
echo "Optional browser automation is not installed automatically; see README for its separate setup."
