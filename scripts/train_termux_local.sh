#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
command -v proot-distro >/dev/null || { echo "Run scripts/setup_termux.sh first."; exit 1; }
proot-distro login debian -- bash -lc '
set -euo pipefail
cd /root/FirstAI
.venv/bin/python -m pip install -q "datasets>=3.0"
.venv/bin/python download_training_data.py --wikitext-out data/raw/wikitext_train.txt --wikitext-config wikitext-2-raw-v1 --max-chars 500000 --skip-instructions
.venv/bin/python prepare_corpus.py
.venv/bin/python merge_instruction_data.py --inputs data/orbit_instructions.example.jsonl examples/real_data/capability_seed_examples.jsonl examples/real_data/seed_examples.jsonl --output data/instructions_local.jsonl --per-source-cap 500
.venv/bin/python train.py --data data/input.txt --out checkpoints/firstgpt-mobile.pt --steps 350 --eval-interval 50 --checkpoint-interval 100 --batch-size 1 --grad-accum-steps 2 --block-size 256 --layers 2 --heads 2 --width 64 --vocab-size 512
.venv/bin/python finetune_instructions.py --checkpoint checkpoints/firstgpt-mobile.best.pt --data data/instructions_local.jsonl --out checkpoints/orbit-mobile-instruct.pt --steps 200 --batch-size 2 --eval-interval 25
echo "Created independent local checkpoint: checkpoints/orbit-mobile-instruct.pt"
'
