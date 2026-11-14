#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA required. Select a Colab T4 runtime.")
print("GPU:",torch.cuda.get_device_name(0))
PY
python -m pip install -q "datasets>=3.0"
python build_all_data.py --country IND --city "New Delhi" --max-per-dataset 3000 --max-base-chars 5000000
python train.py --data data/input.txt --out checkpoints/firstgpt-real.pt --steps "${FIRSTAI_BASE_STEPS:-3500}" --eval-interval 200 --checkpoint-interval 500 --batch-size 4 --grad-accum-steps 4 --block-size 1024 --layers 6 --heads 8 --width 256 --vocab-size 512 --amp
python finetune_instructions.py --checkpoint checkpoints/firstgpt-real.best.pt --data data/instructions_balanced.jsonl --out checkpoints/orbit-instruct.pt --steps "${FIRSTAI_SFT_STEPS:-1800}" --batch-size 4 --eval-interval 100
