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
BASE_ARGS=(--data data/input.txt --out checkpoints/firstgpt-convo.pt --steps "${FIRSTAI_BASE_STEPS:-12000}" --eval-interval 200 --checkpoint-interval 500 --eval-batches 8 --batch-size 2 --grad-accum-steps 8 --block-size 1024 --layers 6 --heads 6 --width 384 --vocab-size 2048 --tokenizer-train-chars 100000 --architecture-version 2 --amp)
if [ -f checkpoints/firstgpt-real.pt ] && [ "${FIRSTAI_RESTART_BASE:-0}" != "1" ]; then python train.py "${BASE_ARGS[@]}" --resume; else python train.py "${BASE_ARGS[@]}"; fi
SFT_ARGS=(--checkpoint checkpoints/firstgpt-convo.best.pt --data data/instructions_balanced.jsonl --out checkpoints/orbit-convo-instruct.pt --steps "${FIRSTAI_SFT_STEPS:-6000}" --batch-size 4 --eval-interval 100 --lr 0.00003)
if [ -f checkpoints/orbit-instruct.pt ] && [ "${FIRSTAI_RESTART_SFT:-0}" != "1" ]; then python finetune_instructions.py "${SFT_ARGS[@]}" --resume; else python finetune_instructions.py "${SFT_ARGS[@]}"; fi
