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
- `orbit.images`: optional ComfyUI-compatible HTTP adapter for a separately hosted image-generation workflow.
- `tests/test_orbit.py`: tests for identity, progress events, fallback execution, and interrupted-task preservation.

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

The runner only executes strategies supplied by the application. A real assistant interface must connect its progress callback to the chat stream or UI and provide real tools. A persistent background worker and automatic resumption scheduler are not included yet.

### Image generation

FirstGPT is still a text-only model. To generate images, install and run a compatible ComfyUI server, load an image model and workflow, then pass the workflow dictionary to `ComfyUIImageBackend`. The adapter does not download models, host ComfyUI, or create an image model on its own.

### Capability and safety note

The identity blueprint supports natural, nonjudgmental discussion of mature topics. It is not a content-filter bypass or a guarantee of unrestricted output. Any deployment that serves adult audiences needs age controls, consent and privacy safeguards, and appropriate content restrictions. The current implementation does not provide an explicit-content image model.

## Scope and limitations

The GPT training foundation is complete as an engineering baseline, and the initial Orbit runtime foundations are now present. This does not mean the model has learned useful language or that a complete autonomous assistant exists. Quality depends on corpus volume and quality, compute, model scale and subsequent training. Full instruction tuning, integrated tools, a chat UI/streaming server, a persistent worker/scheduler, user memory management, and deployment of a separate image-generation model remain future engineering work.
