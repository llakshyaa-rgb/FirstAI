#!/data/data/com.termux/files/usr/bin/bash
# Continue the existing compact FirstGPT run, then instruction-tune it.
# Run from Termux: bash scripts/continue_train_termux.sh
set -euo pipefail
command -v proot-distro >/dev/null || { echo "Install proot-distro first."; exit 1; }

proot-distro login debian -- bash -s <<'DEBIAN'
set -euo pipefail
cd /root/FirstAI

PY=.venv/bin/python
BASE=checkpoints/firstgpt-mobile.pt
BEST=checkpoints/firstgpt-mobile.best.pt
SFT_DATA=data/instructions_local.jsonl
SFT_OUT=checkpoints/orbit-mobile-instruct.pt

for path in "$BASE" "$BEST" data/input.txt "$SFT_DATA"; do
  if [ ! -s "$path" ]; then
    echo "Required file is missing or empty: $path"
    echo "Do not start a new run yet; preserve the existing checkpoints and report this path."
    exit 2
  fi
done

echo "=== FirstGPT preflight ==="
$PY - <<'PY'
import json
from pathlib import Path
p=Path("data/instructions_local.jsonl")
rows=[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"Base corpus: {Path('data/input.txt').stat().st_size:,} bytes")
print(f"Instruction examples: {len(rows):,}")
print(f"Base checkpoint: {Path('checkpoints/firstgpt-mobile.pt').stat().st_size:,} bytes")
print(f"Best checkpoint: {Path('checkpoints/firstgpt-mobile.best.pt').stat().st_size:,} bytes")
if len(rows)<10:
    raise SystemExit("Need at least 10 valid instruction rows; refusing to run SFT.")
PY

CAN_RESUME=$($PY - <<'PY'
import hashlib
import torch
from pathlib import Path
saved=torch.load("checkpoints/firstgpt-mobile.pt",map_location="cpu",weights_only=True)
digest=hashlib.sha256(Path("data/input.txt").read_bytes()).hexdigest()
print("yes" if saved.get("corpus_sha256")==digest else "no")
PY
)

if [ "$CAN_RESUME" = "yes" ]; then
  echo "=== Continue base pretraining for 2,000 more optimizer steps ==="
  $PY train.py \
    --data data/input.txt \
    --out "$BASE" \
    --resume \
    --steps 2000 \
    --eval-interval 100 \
    --checkpoint-interval 250 \
    --batch-size 1 \
    --grad-accum-steps 2 \
    --block-size 256 \
    --layers 2 \
    --heads 2 \
    --width 64 \
    --vocab-size 512 \
    --architecture-version 2
else
  echo "Corpus hash differs from the existing base checkpoint; safely skipping base resume."
  echo "The existing best base checkpoint will still receive corrected instruction tuning."
fi

echo "=== Instruction-tune with inference-aligned prompts ==="
$PY finetune_instructions.py \
  --checkpoint "$BEST" \
  --data "$SFT_DATA" \
  --out "$SFT_OUT" \
  --steps 1200 \
  --batch-size 1 \
  --eval-interval 100 \
  --lr 0.00005

echo "=== Verify the instruction checkpoint ==="
$PY - <<'PY'
import torch
from pathlib import Path
p=Path("checkpoints/orbit-mobile-instruct.pt")
if not p.is_file():
    raise SystemExit("SFT did not create the expected checkpoint.")
data=torch.load(p,map_location="cpu",weights_only=True)
required=("model","config","tokenizer","instruction_data_sha256","instruction_tuning_steps")
missing=[key for key in required if key not in data]
if missing:
    raise SystemExit("Checkpoint is incomplete; missing: "+", ".join(missing))
print("Verified:",p)
print("Base step:",data.get("base_checkpoint_step"))
print("Instruction steps:",data.get("instruction_tuning_steps"))
print("Instruction examples:",data.get("instruction_examples"))
print("Parameters:",sum(t.numel() for t in data["model"].values()))
PY
echo "Training complete. Use checkpoints/orbit-mobile-instruct.pt for Orbit."
DEBIAN
