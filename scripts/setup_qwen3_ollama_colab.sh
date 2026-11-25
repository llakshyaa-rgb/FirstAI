#!/usr/bin/env bash
# Set up the selected Qwen3-8B abliterated GGUF in Colab using Ollama.
# Run from the FirstAI repository root. This does not modify FirstGPT checkpoints.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$PWD/models/qwen3-8b-abliterated-v3.2}"
MODEL_FILE="Qwen3-8B-abliterated-v3.2-Q4_K_M.gguf"
MODEL_NAME="${ORBIT_OLLAMA_MODEL:-firstai-qwen3-8b}"

# Keep compatibility with Colab's datasets stack (which currently requires Hub <2).
python -m pip install -q "huggingface_hub>=0.25,<2"

# Refresh an existing install too: Qwen3's think=false API support requires
# an Ollama release that understands the native chat "think" field.
curl -fsSL https://ollama.com/install.sh | sh

mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL_DIR/$MODEL_FILE" ]; then
  hf download NuevaTechLLC/Qwen3-8B-abliterated-v3.2-GGUF "$MODEL_FILE" --local-dir "$MODEL_DIR"
fi

cat > "$MODEL_DIR/Modelfile" <<EOF
FROM $MODEL_DIR/$MODEL_FILE
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.08
EOF

if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  nohup ollama serve > /tmp/firstai-ollama.log 2>&1 &
fi

for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null

ollama create "$MODEL_NAME" -f "$MODEL_DIR/Modelfile"
echo
echo "Model installed: $MODEL_NAME"
echo "Set these before launching Orbit:"
echo "export ORBIT_LLM_PROVIDER=ollama"
echo "export ORBIT_OLLAMA_MODEL=$MODEL_NAME"
echo "export ORBIT_OLLAMA_URL=http://127.0.0.1:11434"
echo "export ORBIT_CONTEXT_LENGTH=8192"
