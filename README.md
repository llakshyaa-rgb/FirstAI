# FirstAI

FirstAI is a decoder-only language model trained from random initialization.
It does not load pretrained weights or call a hosted language-model API.

## Implemented technical foundation

- Byte-level BPE tokenizer implemented in this repository
- Learned token and positional embeddings
- Multi-head causal self-attention with a future-token mask
- Pre-layer-normalized residual Transformer blocks
- GELU feed-forward networks and tied input/output embeddings
- Next-token cross-entropy objective and AdamW optimization
- Gradient clipping, gradient accumulation, optional CUDA mixed precision, train/validation loss and validation perplexity
- Best-model checkpoints, periodic resumable checkpoints, corpus integrity checks and RNG-state restoration
- Text generation from a saved checkpoint

PyTorch supplies tensor operations, automatic differentiation and device execution.
The model architecture, tokenizer, corpus preparation and training loop are custom.
This is a from-scratch model, not a fine-tune of an existing language model.

## Prepare a corpus

Place UTF-8 `.txt` documents that you have permission to use in `data/raw/`.
Then run:

```bash
python prepare_corpus.py
```

The preparation script normalizes Unicode, removes exact duplicate paragraphs
and writes `data/input.txt`. It does not automatically verify factual quality,
remove every form of boilerplate, or grant rights to source material.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --steps 2000
```

Useful options include `--steps`, `--block-size`, `--layers`, `--heads`,
`--width`, `--batch-size`, `--eval-interval` and `--checkpoint-interval`.
The trainer automatically uses CUDA when available and otherwise CPU.

Resume from the same corpus and checkpoint:

```bash
python train.py --resume --steps 2000
```

The checkpoint stores a SHA-256 fingerprint of the corpus and rejects a changed
corpus rather than silently resuming incompatible training. Saved model
architecture and optimizer state are restored from the checkpoint.

## Evaluate

Evaluate a saved checkpoint against the same corpus used for training:

```bash
python evaluate.py --checkpoint checkpoints/firstgpt.best.pt --data data/input.txt
```

The evaluator checks the corpus fingerprint before reporting train and held-out
validation loss/perplexity.

## Pipeline smoke test

Run a temporary synthetic-data test of training, checkpoint creation, resume,
evaluation and generation without uploading a real corpus:

```bash
python scripts/smoke_test.py
```

## Generate

```bash
python generate.py --checkpoint checkpoints/firstgpt.best.pt --prompt "The " --tokens 200
```

## Tests

```bash
python -m unittest discover -s tests
```

Tests cover BPE UTF-8 round-tripping, unseen characters, corpus deduplication,
next-token batch alignment, model tensor shapes, causal masking, gradient flow
and generation.

## Colab

Open [FirstAI_Colab.ipynb](https://colab.research.google.com/github/llakshyaa-rgb/FirstAI/blob/main/FirstAI_Colab.ipynb).
The notebook checks the active PyTorch/CUDA build and reports the attached GPU before GPU tests. Test 2 deliberately stops if CUDA is unavailable; it does not silently claim a CPU run is a T4 test. Google Drive is optional: if Drive authentication fails, the notebook initializes a temporary local workspace so uploads and training can still proceed during that runtime. Local files are lost when the runtime disconnects, so persistent checkpoints require a successful Drive mount or manual download. A free GPU is not guaranteed.


## Orbit assistant foundations

The repository now includes an initial Orbit runtime layer:

- `ORBIT_IDENTITY.md`: canonical identity and Jarvis-inspired communication contract.
- `orbit.identity`: builds a system prompt from that contract and optional owner/task context.
- `orbit.tasks`: SQLite-backed task/event persistence and a strategy runner that reports a failed method, tries the next supplied strategy, and records completion or failure.
- `orbit.images`: ComfyUI-compatible adapter for a separately hosted image-generation workflow.
- `orbit.memory`: explicit SQLite-backed owner preferences, editable through the local UI/API.
- `orbit.tools`: allow-list tool registry with confirmation gates for consequential actions.
- `orbit.runtime` and `orbit.server`: local chat runtime, HTTP API, browser UI, task progress polling, and restart recovery for interrupted generation tasks.
- `tests/test_orbit.py` and `tests/test_orbit_memory_tools.py`: identity, fallback, persistence, memory, and authorization tests.

Run the tests:

```bash
python -m unittest discover -s tests
```

### Task runner example

```python
from orbit.tasks import TaskStore, TaskRunner

store = TaskStore("orbit_state.sqlite3")
task_id = store.create("Create a project plan")

def primary(goal):
    # Connect a real tool or implementation here.
    return {"plan": f"Plan for: {goal}"}

runner = TaskRunner(store, progress_callback=lambda event: print(event.message))
runner.run(task_id, [("primary workflow", primary)])
```

### Run the local Orbit interface

After training and downloading a compatible checkpoint into the repository:

```bash
python -m orbit.server --checkpoint checkpoints/firstgpt.best.pt --port 8767
```

Open `http://127.0.0.1:8767` on the same device (or use the optional Colab launch cell, which prints a proxied URL). The UI displays live task updates by polling the local event API, and owner preferences are stored in the same SQLite database as task state. Interrupted generation tasks are re-queued on server restart when a usable checkpoint is available. This is restart recovery, not an always-on service: the Python process must remain running.

The tool registry is an allow-list foundation. No arbitrary shell or code execution is exposed, and tools requiring confirmation are blocked until the application supplies explicit confirmation. No external tools are registered by default.


### Orbit local tools and memory

Orbit now has a bounded local tool layer in `orbit/local_tools.py` and a deterministic request router in `orbit/agent.py`. It can calculate arithmetic, list/read/search project text files, create new workspace files, update existing files with timestamped backups, compile-check the project, and run the repository's unit tests. File access is confined to the repository workspace; credential paths, Git internals, backup internals, and symlinks are excluded. It does not expose arbitrary shell commands or Python evaluation.

Examples you can send in the local chat:

- `calculate (144 / 12) + 3`
- `list files`
- `read file README.md`
- `search project for OrbitRuntime`
- `run tests`
- `create file: notes/idea.txt` followed by the file content on the next line
- `update file: notes/idea.txt` followed by replacement content on the next line; the old version is backed up first
- `I prefer concise explanations.` or `remember that I like detailed examples.`
- `what do you remember about me`

Preferences are saved explicitly in the local SQLite memory store and included as context in later model conversations. This is deliberate preference memory, not automatic recording of every private conversation.

The chat now restores saved text history and displays partial generated text while the model is responding. Context budgeting prioritizes the newest owner request over older history and preferences.

### Download open training data without uploads

The optional downloader uses the Hugging Face `datasets` library; it downloads directly into the current workspace and does not require Drive or manual uploads:

```bash
pip install datasets
python download_training_data.py
```

By default it downloads up to 1 million characters from WikiText-2 for base language-model training and Databricks Dolly 15K instruction examples, then appends the Orbit-specific examples. See [data/SOURCES.md](data/SOURCES.md) for dataset sources and attribution. The WikiText dataset card describes over 100 million tokens across its variants; Dolly contains approximately 15,000 human-written instruction/response records. See the [WikiText card](https://huggingface.co/datasets/Salesforce/wikitext) and [Dolly card](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Train a fresh base checkpoint and then instruction-tune it:

```bash
python train.py --data data/wikitext_train.txt --out checkpoints/firstgpt-wikitext.pt --steps 2000 --block-size 256 --layers 6 --heads 8 --width 256 --batch-size 8 --grad-accum-steps 2 --amp
python finetune_instructions.py --checkpoint checkpoints/firstgpt-wikitext.best.pt --data data/orbit_instructions_full.jsonl --out checkpoints/orbit-instruct.pt --steps 1000 --batch-size 8
```

This is an experimental small-model training path. It will improve the training signal over the synthetic smoke-test corpus, but it cannot guarantee strong reasoning, long-form writing, factual reliability, or general assistant performance.

### Orbit instruction tuning

A small supervised fine-tuning script is included:

```bash
python finetune_instructions.py --checkpoint checkpoints/firstgpt.best.pt --data data/orbit_instructions.example.jsonl --out checkpoints/orbit-instruct.pt --steps 1000
```

JSONL examples use `prompt` and `response` fields. The trainer masks prompt tokens so the supervised loss is applied to the response, holds out validation examples, clips gradients, and writes the best validation checkpoint atomically. The included examples are only a format/demo starter set. They are far too small to create a capable assistant; build a large, diverse, carefully reviewed instruction corpus before expecting meaningful instruction-following. The SFT script preserves the base checkpoint and writes a separate output file.

### Image studio

The local UI includes an Image Studio panel. To enable it, run a ComfyUI server with an image model installed and export a ComfyUI API workflow JSON. Configure prompt placeholders `{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, and `{{SEED}}`, then set these environment variables before starting Orbit:

```bash
export ORBIT_COMFYUI_URL=http://127.0.0.1:8188
export ORBIT_IMAGE_WORKFLOW=$PWD/workflows/comfyui_checkpoint_template.json
export ORBIT_IMAGE_CHECKPOINT=your-installed-model.safetensors
export ORBIT_IMAGE_WIDTH=768
export ORBIT_IMAGE_HEIGHT=768
export ORBIT_IMAGE_STEPS=24
export ORBIT_IMAGE_CFG=7.0
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

The UI submits image tasks asynchronously, records progress, saves returned images under `orbit/generated/`, and displays them in the conversation. The image backend is external and is not bundled or automatically downloaded. Its model and workflow determine which visual requests it can fulfill; this code does not bypass model restrictions or guarantee nude/explicit output. The local prototype has no built-in age verification, authentication, or full content-moderation system. Do not expose it to a network or public users as-is. A public adult-facing deployment needs authentication, age assurance, consent/privacy protections, rate limits, and robust content safeguards. See [ORBIT_IMAGE_SETUP.md](ORBIT_IMAGE_SETUP.md).

### Image generation

FirstGPT is still a text-only model. To generate images, install and run a compatible ComfyUI server, load an image model and workflow, then pass the workflow dictionary to `ComfyUIImageBackend`. The adapter does not download models, host ComfyUI, or create an image model on its own.

### Capability and safety note

The identity blueprint supports natural, nonjudgmental discussion of mature topics. It is not a content-filter bypass or a guarantee of unrestricted output. Any deployment that serves adult audiences needs age controls, consent and privacy safeguards, and appropriate content restrictions. The current local image adapter does not bundle an image model; its output depends on the separately configured ComfyUI model and workflow.

## Scope and limitations

The GPT training foundation is complete as an engineering baseline, and the initial Orbit runtime foundations are now present. This does not mean the model has learned useful language or that a complete autonomous assistant exists. Quality depends on corpus volume and quality, compute, model scale and subsequent training. The instruction-tuning pipeline and image-studio adapter/UI are implemented, but the included tuning examples are only a tiny demo. The optional open-corpus and Dolly instruction-data pipeline is implemented but must still be run on a GPU. Local deterministic tools are available, but general model-driven tool planning/execution, vision understanding, robust tool recovery, and extensive end-to-end testing remain future engineering work. Orbit is currently a local prototype, not an always-on service.
