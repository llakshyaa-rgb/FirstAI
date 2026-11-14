#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA required. Select a Colab T4 runtime.")
print("GPU:",torch.cuda.get_device_name(0),"VRAM GiB:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
PY
python -m pip install -q "datasets>=3.0"
python build_all_data.py --country "${FIRSTAI_COUNTRY:-IND}" --city "${FIRSTAI_CITY:-New Delhi}" --max-per-dataset "${FIRSTAI_MAX_PER_DATASET:-10000}" --max-base-chars "${FIRSTAI_MAX_BASE_CHARS:-30000000}" --per-source-cap "${FIRSTAI_PER_SOURCE_CAP:-10000}"
BASE_ARGS=(--data data/input.txt --out checkpoints/firstgpt-real.pt --steps "${FIRSTAI_BASE_STEPS:-10000}" --eval-interval 200 --checkpoint-interval 250 --eval-batches 12 --batch-size 4 --grad-accum-steps 4 --block-size 1024 --layers 6 --heads 8 --width 256 --vocab-size 512 --amp)
if [ -f checkpoints/firstgpt-real.pt ] && [ "${FIRSTAI_RESTART_BASE:-0}" != "1" ]; then python train.py "${BASE_ARGS[@]}" --resume; else python train.py "${BASE_ARGS[@]}"; fi
SFT_ARGS=(--checkpoint checkpoints/firstgpt-real.best.pt --data data/instructions_balanced.jsonl --out checkpoints/orbit-instruct.pt --steps "${FIRSTAI_SFT_STEPS:-5000}" --batch-size 4 --eval-interval 100)
if [ -f checkpoints/orbit-instruct.pt ] && [ "${FIRSTAI_RESTART_SFT:-0}" != "1" ]; then python finetune_instructions.py "${SFT_ARGS[@]}" --resume; else python finetune_instructions.py "${SFT_ARGS[@]}"; fi
