#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
command -v proot-distro >/dev/null || { echo "Run scripts/setup_termux.sh first."; exit 1; }
proot-distro login debian -- bash -lc '
set -euo pipefail
cd /root/FirstAI
.venv/bin/python -m pip install -q "datasets>=3.0"
.venv/bin/python download_training_data.py --wikitext-out data/raw/wikitext_train.txt --wikitext-config wikitext-2-raw-v1 --max-chars 1000000 --skip-instructions
.venv/bin/python download_capability_data.py --max-per-dataset 1000
.venv/bin/python examples/real_data/build_data_lab.py --country IND --city "New Delhi"
.venv/bin/python examples/real_data/build_training_examples.py
.venv/bin/python prepare_corpus.py
.venv/bin/python merge_instruction_data.py --inputs data/orbit_instructions.example.jsonl data/instructions_capabilities.jsonl examples/real_data/capability_seed_examples.jsonl examples/real_data/seed_examples.jsonl examples/real_data/generated/data_lab_qa_examples.jsonl --output data/instructions_local.jsonl --per-source-cap 1000
.venv/bin/python train.py --data data/input.txt --out checkpoints/firstgpt-mobile.pt --steps 500 --eval-interval 50 --checkpoint-interval 100 --batch-size 1 --grad-accum-steps 2 --block-size 256 --layers 2 --heads 2 --width 64 --vocab-size 512 --architecture-version 2
.venv/bin/python finetune_instructions.py --checkpoint checkpoints/firstgpt-mobile.best.pt --data data/instructions_local.jsonl --out checkpoints/orbit-mobile-instruct.pt --steps 300 --batch-size 1 --eval-interval 25
echo "Created independent local checkpoint: /root/FirstAI/checkpoints/orbit-mobile-instruct.pt"
echo "Compact CPU profile; Android GPU acceleration remains a separate backend task."
'
