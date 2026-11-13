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
The notebook checks the active PyTorch/CUDA build and reports the attached GPU before GPU tests. Test 2 deliberately stops if CUDA is unavailable; it does not silently claim a CPU run is a T4 test. The workflow uses temporary Colab storage only: no Drive mount and no manual data uploads are required. Runtime files disappear when Colab disconnects, so download checkpoints before ending the session. A free GPU is not guaranteed.


## Orbit assistant foundations

The repository now includes an initial Orbit runtime layer:

- `ORBIT_IDENTITY.md`: canonical identity and Jarvis-inspired communication contract.
- `orbit.identity`: builds a system prompt from that contract and optional owner/task context.
- `orbit.tasks`: SQLite-backed task/event persistence and a strategy runner that reports a failed method, tries the next supplied strategy, and records completion or failure.
- `orbit.planner`: strict JSON tool-plan validation; model-generated plans can only call registered read-only tools, never arbitrary shell or write operations.
- `orbit.agents`: a coordinator and parallel researcher, workspace-inspector and calculation specialists. They share the existing runtime and execute independent read-only tools concurrently; they do not load duplicate LLM weights or make extra model calls.
- `orbit.web_search`: read-only Brave web research with source links.
- `orbit.images`: ComfyUI-compatible adapter for a separately hosted image-generation workflow.
- `orbit.media`: private local uploads, format validation, text extraction, upload library, and Brave image search with a mature-results SafeSearch toggle.
- `orbit.vision`: Ollama and OpenAI-compatible vision adapters for actual image-pixel analysis.
- `orbit.media_processing`: optional scanned-PDF OCR, ZIP text inspection, audio transcription and sampled-frame video understanding.
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

### Orbit agent team and hardware budget

Orbit uses a coordinator with five lightweight specialists:
- **Researcher:** public web search and source gathering.
- **Workspace inspector:** project/file search.
- **Analyst:** restricted arithmetic and numerical checks.
- **Media specialist:** processes uploaded documents, images, archives, audio and video, using safe text fallbacks when a parser or vision backend is unavailable.
- **Verifier:** runs compilation and tests when explicitly asked to verify the project.

Independent specialist tools run concurrently (default maximum: 3 workers). The main language model is shared, so the agents do not each load another model onto the GPU. The model synthesizes the collected results once. This is deliberately a tool-specialist team, not three duplicate LLMs; separate model instances would consume more RAM/VRAM and often slow a phone down.

Inference defaults are conservative and configurable:

```bash
export ORBIT_GPU_MEMORY_FRACTION=0.86
export ORBIT_CPU_THREADS=4
export ORBIT_AGENT_WORKERS=3
export ORBIT_MAX_NEW_TOKENS=120
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

The GPU setting is a **per-process PyTorch CUDA allocator ceiling**, not a command to keep GPU utilisation at 86%. It limits how much of the selected CUDA GPU's memory this Orbit process may allocate. It does not reserve or guarantee that amount, and other GPU users (including separate vision/image services) have their own memory use. The ceiling is clamped between 50% and 90%; the default is 86%. Mixed-precision inference (BF16 when supported, otherwise FP16) and TF32 matmul are enabled on CUDA. One shared model lock prevents simultaneous chat generations from overloading the device; specialist I/O and file checks can still run in parallel.

On a phone, system RAM is shared with Android and its graphics stack. The CUDA memory setting only applies when PyTorch actually detects a CUDA GPU; it does not make Android GPUs CUDA-compatible. Termux devices without CUDA run the model on CPU, where `ORBIT_CPU_THREADS` limits PyTorch's CPU thread count. A 4-thread default leaves more CPU capacity for Android and browser tasks, but speed still depends on the model size and phone hardware. For heavy video/vision work, use the remote Colab T4 backend rather than trying to keep large models resident on an 8 GB phone.

### Diagnose the local setup

Before launching Orbit, run:

```bash
python -m orbit.doctor --checkpoint checkpoints/orbit-instruct.pt
```

The report checks the language checkpoint, CUDA availability, optional browser/document/audio packages, and configured image/vision services. It never prints API-key values. Missing optional services do not block text-only Orbit.

### Run the local Orbit interface

After training and downloading a compatible checkpoint into the repository:

```bash
python -m orbit.server --checkpoint checkpoints/firstgpt.best.pt --port 8767
```

Open `http://127.0.0.1:8767` on the same device (or use the optional Colab launch cell, which prints a proxied URL). The UI displays live task updates by polling the local event API, and owner preferences are stored in the same SQLite database as task state. Interrupted generation tasks are re-queued on server restart when a usable checkpoint is available. This is restart recovery, not an always-on service: the Python process must remain running.

The allow-list registry is connected to Orbit's local calculator, workspace file tools, compiler check, test runner and optional Brave web search. A coordinator assigns relevant researcher, workspace, analyst, verifier and media specialists. Independent tasks run concurrently within a bounded worker pool; the media specialist handles owner attachments while research and workspace inspection can proceed in parallel. Results and failures are passed to the main assistant for synthesis. Agents share the same language runtime rather than loading duplicate models. This is a tool-using multi-agent coordinator, not yet a fully autonomous team of independent code-writing LLMs. Specialist tool calls are bounded and read-only; they do not spawn separate copies of the language model. A constrained tool-planning interface lets FirstGPT propose multi-step read-only tool calls; every proposed action is validated against the allow-list, and writes or code execution are excluded from model-generated plans. File creation/updates and test execution are only routed after an explicit owner command. No arbitrary shell or unrestricted code execution is exposed.


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

The chat now restores saved text history and displays partial generated text while the model is responding. Generation preserves multi-line responses instead of stopping at the first newline. Context budgeting prioritizes the newest owner request over older history and preferences.

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

Orbit now has two image-generation backends:

- **Diffusers (direct GPU backend):** no ComfyUI server or workflow JSON is required. Install requirements-image.txt, set ORBIT_IMAGE_BACKEND=diffusers, and choose a public Diffusers model with ORBIT_IMAGE_MODEL. The first generation downloads model weights automatically.
- **ComfyUI:** retain the existing workflow adapter for custom graphs and externally hosted image models by setting ORBIT_IMAGE_BACKEND=comfyui, ORBIT_COMFYUI_URL, and ORBIT_IMAGE_WORKFLOW.

The Colab notebook includes an optional T4 setup cell for Diffusers using Lykon/dreamshaper-8. It downloads the model on first use. During Diffusers generation, Orbit temporarily moves FirstGPT to CPU, clears its CUDA allocations, renders with Diffusers CPU offload, and restores FirstGPT to its original device. This reduces GPU-memory overlap; it does not make image generation instantaneous or guarantee compatibility with every GPU. The UI submits image jobs asynchronously and displays saved results in the conversation.

For a local Diffusers setup:

~~~bash
pip install -r requirements-image.txt
export ORBIT_IMAGE_BACKEND=diffusers
export ORBIT_IMAGE_MODEL=Lykon/dreamshaper-8
export ORBIT_IMAGE_DEVICE=auto
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
~~~

Use ORBIT_IMAGE_WIDTH, ORBIT_IMAGE_HEIGHT, ORBIT_IMAGE_STEPS, and ORBIT_IMAGE_CFG to tune generation. The direct backend defaults to 512×512 and 20 steps. FirstGPT remains a text model; pixels are generated by the separate diffusion model. Model licences and model-specific output restrictions still apply.

### Images, files and image search

Uploads are limited to 20 MB per batch. Orbit previews common images (JPG/JPEG, PNG, WEBP, GIF, BMP, TIFF, AVIF and HEIC/HEIF) and can pass their actual pixels to a configured vision model. FirstGPT itself has not been trained with a vision encoder.

Common readable formats include:
- Text, Markdown, CSV, JSON, XML, HTML, RTF and source-code files.
- PDF, DOCX, XLSX, PPTX and OpenDocument files (ODT, ODS, ODP).
- EPUB books.
- ZIP, TAR and TGZ archives. Orbit inspects supported text entries without extracting or executing archive contents.
- Audio formats such as MP3, WAV, M4A, OGG and FLAC; video formats such as MP4, MOV, WEBM and MKV.

PDF text extraction uses pypdf; scanned-page OCR uses optional PDFium plus the configured vision model. Audio transcription uses optional faster-whisper. Video analysis samples up to six frames and, when ffmpeg and faster-whisper are available, transcribes up to five minutes of its audio track. This is sampled media understanding, not frame-by-frame temporal comprehension. Optional media dependencies are in requirements-media.txt; ffmpeg is installed separately.

Other file formats can still be safely stored and downloaded, but Orbit does not claim to understand a format without a parser. Legacy binary Office formats (DOC/XLS/PPT) and proprietary formats are not currently parsed.

Internet image search uses the Brave Search API. Set BRAVE_SEARCH_API_KEY; the mature-results toggle selects SafeSearch off or strict. Search returns thumbnails and original source-page links. Availability of mature results depends on the provider. Respect source-site access controls and image licences.

The optional Colab vision cell installs Ollama and downloads LLaVA 7B for image reading. The image and vision backends are separate models; they are not learned capabilities of FirstGPT. Do not send private images to a remote endpoint you do not trust.

### Public website browser

Orbit includes an optional Playwright browser adapter. Install it with:

```bash
pip install -r requirements-browser.txt
python -m playwright install chromium
```

Explicit browser commands in Orbit's local chat:

- `open website: https://example.com` — navigate and read visible page text.
- `read website` — read the currently open page.
- `fill website: input[name=q]` followed by the text on the next line — fill a field without submitting it.
- `click website: text=Continue` — click the selector you explicitly specify; this can change website state.

The browser is optional and starts only when used. It restricts navigation and page requests to public HTTP(S) addresses and blocks localhost/private-network destinations. It does not bypass logins, CAPTCHAs, paywalls, website restrictions, or account permissions. Form submission, purchases, and other consequential website actions are not automated.

The Colab notebook includes optional browser, vision and direct Diffusers image-generation setup cells. When the separate Ollama vision model or Diffusers image backend is enabled, the Orbit PyTorch CUDA allocator ceiling is lowered to 60% to leave more VRAM headroom; otherwise it remains 86%. This is a memory-allocation ceiling, not a GPU-utilisation target.

### Capability and safety note

The identity blueprint supports natural, nonjudgmental discussion of mature topics within applicable boundaries. The local image adapter does not bundle an image model; its output depends on the separately configured ComfyUI model and workflow.

## Scope and limitations

The GPT training foundation is complete as an engineering baseline, and the initial Orbit runtime foundations are now present. This does not mean the model has learned useful language or that a complete autonomous assistant exists. Quality depends on corpus volume and quality, compute, model scale and subsequent training. The instruction-tuning pipeline and image-studio adapter/UI are implemented, but the included tuning examples are only a tiny demo. The optional open-corpus and Dolly instruction-data pipeline is implemented but must still be run on a GPU. Local deterministic tools are available, but open-ended general-purpose tool planning and execution remain future engineering work; the current model planner is deliberately limited to validated read-only tools. Public-site browser commands are available through the optional Playwright adapter, but they are explicit selector-based actions rather than autonomous account workflows. Native vision training inside FirstGPT, full audio/video comprehension, robust recovery across every external service, and extensive end-to-end testing also remain future work. Orbit's connected vision and optional media-processing adapters require their respective external/local backends. Orbit is currently a local prototype, not an always-on service.
