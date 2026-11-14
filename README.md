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



## Capability-wide real data

FirstAI now has a source registry at `data/real_world_sources.json`. It maps each capability to its public data/training sources, licenses, provenance policy and current integration status.

### One-command complete data preparation

For the full pipeline, install the optional Hugging Face dataset library and run:

```bash
pip install datasets
python build_all_data.py
```

This downloads the base WikiText/Dolly data, capability datasets, collects the seven-domain live-data lab, generates grounded examples, prepares `data/input.txt`, and creates a deduplicated, per-source-balanced `data/instructions_balanced.jsonl`. It writes source manifests and stops on a failed stage without deleting earlier outputs.

You can skip individual download stages when you already have their files:

```bash
python build_all_data.py --skip-base-text
python build_all_data.py --skip-capability-datasets
python build_all_data.py --skip-live-data
```

The pipeline prepares data only. It does not train or replace a model checkpoint.
### Download real training examples for more skills

```bash
pip install datasets
python download_capability_data.py --max-per-dataset 3000
```

The repository already contains small real sourced seed examples across economics, mathematics, science, reading comprehension and coding. The larger download adds up to 3,000 examples per source by default. This adds real training examples for:
- Mathematics: GSM8K worked word problems.
- Science: AI2 ARC Challenge questions.
- Reading comprehension: SQuAD passage-grounded answers.
- Coding: sanitized MBPP programming tasks and tests.

It saves per-dataset JSONL, a combined `data/instructions_capabilities.jsonl` file and a provenance manifest. Only training splits are used; benchmark test splits remain excluded.

Instruction-tune a checkpoint with the new data:

```bash
python finetune_instructions.py --checkpoint checkpoints/firstgpt.best.pt --data data/instructions_balanced.jsonl --out checkpoints/firstgpt-capabilities.pt
```

### Orbit's additional live knowledge

Orbit's `knowledge_search` tool now routes encyclopedic questions to Wikipedia, scholarly discovery to OpenAlex/arXiv, and programming questions to Stack Exchange. It includes source links, retrieval times, license notes and caveats. Image search has a strict-safe Wikimedia Commons fallback with per-file attribution metadata when Brave is not configured.

The existing live-data APIs remain separate from training. Real data is not automatically written into model weights; collect, inspect, prepare and train deliberately. Audio/video, image understanding, generated media and private memory remain distinct external or private-data capabilities rather than being misrepresented as text training.

## Real-world training examples

The starter corpus now includes sourced India examples in `data/raw/real_world_examples.txt` and mathematics, science, reading-comprehension and coding seed examples in `data/raw/capability_seed_examples.txt`; `prepare_corpus.py` includes this file automatically. The matching structured examples, with source links, indicator IDs, years, units and retrieval dates, are in `examples/real_data/seed_examples.jsonl`.

For a larger, refreshed multi-domain set, run:

```bash
python examples/real_data/build_data_lab.py
python examples/real_data/build_training_examples.py
python prepare_corpus.py
```

This collects real records from seven public-data domains and creates a training text file plus an auditable JSONL companion. Forecasts, historical observations and reference rates are labeled distinctly. A data refresh does not change model weights: train or resume FirstGPT separately after reviewing the corpus.

## Real-world data access

Orbit now has a read-only live-data specialist connected to the single chat interface. Supported questions are automatically routed to public structured APIs for:
- Weather and short-range forecasts (Open-Meteo)
- Air-quality model forecasts (Open-Meteo)
- Magnitude 4.5+ earthquakes from the past day (USGS)
- Currency reference rates (Frankfurter)
- Open natural-event catalog (NASA EONET)
- Country reference profiles (REST Countries)
- Annual economic/development indicators (World Bank)

Example prompts: `weather in Patna`, `air quality in Delhi`, `India GDP`, `Japan inflation`, `USD to INR exchange rate`, `latest earthquakes`, or `country profile for India`.

Responses include the provider, source URL, retrieval timestamp, available units and important caveats. Forecasts are labeled as forecasts, exchange rates as reference rates, and country profiles as reference metadata. These live lookups require network access from the machine running Orbit but no API key for the listed endpoints. General web research remains a separate capability and may require `BRAVE_SEARCH_API_KEY`.

**Live data is not model training.** Orbit fetches these records when a supported question is asked and passes a compact evidence summary to FirstGPT. The downloaded records are not automatically written into the model's weights or corpus.

Build a full local data pack:

```bash
python examples/real_data/build_data_lab.py
python examples/real_data/validate_data_lab.py
```

Create a timestamped refresh and compare it with the previous snapshot:

```bash
python examples/real_data/refresh_data_lab.py
```

The refresh utility retains historical snapshots, source logs, validation output and a source-by-source change report under `examples/real_data/generated/data_lab_history/`. It does not overwrite earlier snapshots.

### Inspect a running app in a real browser

The local preview panel now includes **Inspect app**. It uses the optional Playwright Chromium backend to open the running loopback preview and report its page title, headings, buttons, links, form fields, image loading, JavaScript console errors and failed requests. It does not click buttons or submit forms. Install it when you want this browser inspection:

```bash
pip install -r requirements-browser.txt
python -m playwright install chromium
```

The iframe remains the interactive place where you can manually use the app. Browser inspection is read-only and restricted to the exact local preview host and port. Cloud preview still does not support WebSocket/HMR forwarding; refresh the cloud preview manually after changes.


## Orbit assistant foundations

The repository now includes an initial Orbit runtime layer:

- `ORBIT_IDENTITY.md`: canonical identity and Jarvis-inspired communication contract.
- `orbit.identity`: builds a system prompt from that contract and optional owner/task context.
- `orbit.tasks`: SQLite-backed task/event persistence and a strategy runner that reports a failed method, tries the next supplied strategy, and records completion or failure.
- `orbit.planner`: strict JSON tool-plan validation; model-generated plans can only call registered read-only tools, never arbitrary shell or write operations.
- `orbit.intent` and `orbit.agents`: one automatic capability router and bounded specialist team. Researcher, image researcher, code scout, debugger, reviewer, analyst, verifier and media agents are selected from the request; relevant independent work runs concurrently and Orbit returns one combined chat response. Coding agents build a bounded AST-based project map and trace relevant source references before suggesting changes.
- `orbit.web_search`: read-only Brave web research with source links.
- `orbit.images`: ComfyUI-compatible adapter for a separately hosted image-generation workflow.
- `orbit.media`: local uploads, format validation, text extraction, upload library, explicit upload deletion, and Brave image search with a user-controlled SafeSearch toggle. Local storage is not an access-control system; do not expose the prototype to untrusted networks.
- `orbit.vision`: Ollama and OpenAI-compatible vision adapters for actual image-pixel analysis.
- `orbit.media_processing`: optional scanned-PDF OCR, ZIP text inspection, audio transcription and sampled-frame video understanding.
- `orbit.memory`: explicit SQLite-backed owner preferences, editable through the local UI/API.
- `orbit.tools`: allow-list tool registry with confirmation gates for consequential actions.
- `orbit.runtime` and `orbit.server`: local chat runtime, HTTP API, browser UI, task progress polling, restart recovery for interrupted chat/design jobs, a bounded background worker pool, and in-chat retry for failed or interrupted jobs.
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

Independent specialist tools run concurrently (default maximum: 3 workers). The main language model is shared, so the agents do not each load another model onto the GPU. The model synthesizes the collected results once. The media specialist receives attachment IDs directly and processes them while unrelated research or workspace tasks run concurrently. This is deliberately a tool-specialist team, not three duplicate LLMs; separate model instances would consume more RAM/VRAM and often slow a phone down.

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

Open `http://127.0.0.1:8767` on the same device (or use the optional Colab launch cell, which prints a proxied URL). The UI displays live task updates by polling the local event API, and owner preferences are stored in the same SQLite database as task state. Interrupted chat and design tasks are re-queued on server restart when a usable checkpoint is available; image jobs preserve their prompt and can be retried from chat. Failed or interrupted jobs expose an in-chat retry action. Jobs run through a bounded worker pool. This is restart recovery, not an always-on service: the Python process must remain running.

Set `ORBIT_WORKSPACE_ROOT` to a user-selected local repository path when Orbit should work on a project other than FirstAI. The coordinator selects researcher, image researcher, workspace scout, debugger, reviewer, analyst, verifier and media specialists as needed. Independent work runs concurrently using one shared model runtime. Explicit software implementation requests enter the autonomous coding workflow: inspect the project, decompose work into up to six phases with two target files per phase, assign architect, designer, frontend, backend, data, security or QA responsibilities to each phase, propose exact patches, run an independent review pass, preserve timestamped backups, compile-check Python, syntax-check changed JavaScript when Node is available, and attempt up to two diagnose-and-repair rounds. Each phase now gets its own durable role-scoped workspace under `.orbit_agent_workspaces/`, including a separate on-disk snapshot of its allowed project files. The reviewer gets a separate snapshot and review folder; briefs, proposals, reviews and applied-change records are retained. The lead is the only project writer, so specialist outputs cannot race to overwrite one another. These are isolated task workspaces and prompts coordinated through the shared FirstGPT runtime, not separate model instances or OS-level sandboxes; the two-file phase limit keeps patches bounded for the current small context window. Existing files are changed only with exact unique patch anchors; ambiguous patches are rejected before any file in that batch is written. Credential-bearing paths and files that appear to contain secrets are excluded. No arbitrary shell command execution is exposed. Project build scripts and full test suites are not run by default because they execute project code. After reviewing and trusting the target workspace, opt in with `ORBIT_AUTORUN_BUILD=true` to run the nearest changed app's `npm/pnpm/yarn/bun run build` script (when present), and `ORBIT_AUTORUN_TESTS=true` to run the repository test suite. Build checks are bounded to 240 seconds and their output is fed into the repair loop. These opt-ins are not a security sandbox; never enable them for untrusted projects.



### Real-data example pack

The repository includes two real-data builders under `examples/real_data/`. The focused World Bank example is `python examples/real_data/build_examples.py`. The multi-domain data lab is `python examples/real_data/build_data_lab.py`, which independently retrieves seven datasets from six public-data providers: World Bank economic indicators, Open-Meteo weather and air-quality forecasts, USGS magnitude 4.5+ earthquakes, Frankfurter INR reference rates, NASA EONET natural events and REST Countries country profiles. It writes source-attributed JSON/CSV files, a manifest, and a responsive HTML dashboard; partial provider failures do not discard successful datasets. Default country is India (IND), default city is New Delhi; use `--country`, `--city`, `--start-year` and `--end-year` to change inputs. Missing values remain null/blank, not zero. Run `python examples/real_data/validate_data_lab.py` for an offline JSON/CSV/provenance integrity check. These are live API adapters; the real data is fetched when you run the builder, not committed as fabricated sample values.

`benchmark_tasks.json` contains concrete engineering scenarios for each data domain, integration, data quality, research, spreadsheets, debugging, security and reliability. Acceptance criteria are instructions for later evaluation, not claims that tasks have passed. See [the example guide](examples/real_data/README.md) for sources, commands and caveats.

### Live application preview

Orbit's single chat interface now includes a **Preview app** panel. Set `ORBIT_WORKSPACE_ROOT` to the workspace root, ask Orbit to build or modify the app, and press Preview app to launch the project's `package.json` `dev` script (or a static `index.html` preview) and interact with the actual running site in an embedded frame. If the runnable app is inside a subfolder, Orbit attempts to detect it from the files changed by the coding task and fills the preview folder field; you can also enter a relative path manually (for example, `apps/site`). Parent traversal, symlinks and protected folders are rejected. You can open it in a separate browser tab or stop the preview. The preview server binds to loopback only; static previews block hidden files, credential-like dotfiles, dependency folders and symlink escapes. Starting a package script executes that project's own development command, so only preview code from a workspace you trust. This is a local development preview, not production deployment or an app-store build. On Android, the project must have Node and its package manager available in Termux. For a remote Colab/runtime project, expose its app through a public HTTPS tunnel (for example, a Cloudflare Tunnel URL) and set `ORBIT_REMOTE_PREVIEW_URL=https://your-tunnel.example` in the environment where Orbit's server runs. Example flow in Colab: start the app on port 4173, expose `http://127.0.0.1:4173` with your tunnel client, copy its HTTPS URL, then restart Orbit with that environment variable set in Termux. Orbit proxies HTTP GET/POST requests through `/remote-preview/` into that fixed HTTPS origin and displays it in an isolated loopback proxy on a separate port/origin, so the remote page cannot read Orbit's chat page or API responses. This proxy is intended for static/production-style frontends; configure the remote app's asset base and API paths under `/remote-preview/` for best compatibility. WebSockets/HMR and arbitrary cross-origin backend paths are not proxied, so refresh manually and keep APIs same-origin under the proxy path. Native HTML form submission is disabled in the cloud iframe; use same-origin JavaScript requests instead. Example setup:

- In the Colab runtime, start the app on a port such as 4173, bound to `0.0.0.0` or `127.0.0.1` as required by the tunnel client.
- Start the tunnel to that port, for example `cloudflared tunnel --url http://127.0.0.1:4173`, and copy the generated HTTPS URL.
- In the Termux shell that launches Orbit, set `export ORBIT_REMOTE_PREVIEW_URL="https://your-generated-host"` and restart Orbit.
- Open Orbit through `http://127.0.0.1:8767` or `http://localhost:8767` so the cloud iframe can use the alternate loopback origin.

The generated tunnel URL may change when the Colab runtime restarts.

Keep the tunnel private to your own work and do not expose sensitive development servers publicly.

The health endpoint reports the loaded checkpoint's actual `context_window_tokens`. This is the model's trained positional context, not the number of chat messages. Conversation history is persisted in SQLite, but the current small model prompt builder can only fit a limited slice of it. Longer chats therefore require context-window scaling and rolling summaries/retrieval; saved chat history alone does not mean the model can attend to every old message at once.

### Orbit local tools and memory

Orbit now has a bounded local tool layer in `orbit/local_tools.py` and a deterministic request router in `orbit/agent.py`. It can calculate arithmetic, list/read/search project text files, create new workspace files, update existing files with timestamped backups, compile-check the project, and run repository unit tests when explicitly enabled. File access is confined to the repository workspace; credential paths, Git internals, backup internals, and symlinks are excluded. It does not expose arbitrary shell commands or Python evaluation.

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

By default it downloads up to 5 million characters from WikiText-103 for base language-model training and Databricks Dolly 15K instruction examples, then appends Orbit-specific examples covering general assistant behaviour, natural mature conversation, consent, and sensitive-media handling. See [data/SOURCES.md](data/SOURCES.md) for dataset sources and attribution. The WikiText dataset card describes over 100 million tokens across its variants; Dolly contains approximately 15,000 human-written instruction/response records. See the [WikiText card](https://huggingface.co/datasets/Salesforce/wikitext) and [Dolly card](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

Train a fresh base checkpoint and then instruction-tune it:

```bash
python train.py --data data/wikitext_train.txt --out checkpoints/firstgpt-wikitext.pt --steps 4000 --block-size 256 --layers 6 --heads 8 --width 256 --batch-size 8 --grad-accum-steps 2 --amp
python finetune_instructions.py --checkpoint checkpoints/firstgpt-wikitext.best.pt --data data/orbit_instructions_full.jsonl --out checkpoints/orbit-instruct.pt --steps 2000 --batch-size 8
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

The Colab notebook includes an optional T4 setup cell for Diffusers using Lykon/dreamshaper-8. It downloads the model on first use. During Diffusers generation, Orbit temporarily moves FirstGPT to CPU, clears its CUDA allocations, renders with Diffusers CPU offload, and restores FirstGPT to its original device. This reduces GPU-memory overlap; it does not make image generation instantaneous or guarantee compatibility with every GPU. The UI submits image jobs asynchronously and displays saved results in the conversation. You can also type `create image: ...` or `generate image: ...` directly into the main chat composer.

For a local Diffusers setup:

~~~bash
pip install -r requirements-image.txt
export ORBIT_IMAGE_BACKEND=diffusers
export ORBIT_IMAGE_MODEL=Lykon/dreamshaper-8
export ORBIT_IMAGE_DEVICE=auto
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
~~~

Use ORBIT_IMAGE_WIDTH, ORBIT_IMAGE_HEIGHT, ORBIT_IMAGE_STEPS, and ORBIT_IMAGE_CFG to tune generation. The direct backend defaults to 512×512 and 20 steps. FirstGPT remains a text model; pixels are generated by the separate diffusion model. Model licences and model-specific output restrictions still apply.

### Creative Studio — create real deliverable files

Orbit includes an AI-assisted design workflow and a multi-format export engine. Describe the deliverable in the main chat composer; Orbit automatically chooses a supported output format. Orbit asks the trained FirstGPT checkpoint to draft the content, formats it and saves a downloadable artifact under orbit/generated/. You can also start from the main chat with explicit requests such as `create a PDF about ...`, `make a presentation about ...`, `build a website for ...`, or `design a poster about ...`.

Supported output formats:
- **PDF** — paginated A4 documents with headings, tables, lists, quotes, code blocks, page numbers and a restrained footer.
- **DOCX** — editable Microsoft Word documents.
- **PPTX** — PowerPoint slide decks with title and content slides.
- **XLSX** — Excel workbooks with headings, tables, styled header rows and adjusted columns.
- **CSV and JSON** — structured table/data and block-structured document export.
- **HTML** — responsive, styled standalone web pages.
- **SVG** — scalable vector poster/one-page graphic layouts.
- **Markdown and TXT** — editable source documents.

Four built-in themes are available: Midnight, Paper, Ocean and Editorial. The same Markdown content structure is rendered into the selected format. Image-heavy artwork can be created separately through the Image Studio and placed into a document in a future layout iteration.

The exporter dependencies are optional. Install them with `pip install -r requirements-design.txt` (or run the optional Creative Studio setup cell in Colab). HTML, SVG, Markdown, CSV and TXT need only the standard library. The AI design workflow requires a loaded, trained FirstGPT checkpoint; without one, Orbit reports that it cannot draft the content rather than returning an empty file. Export code is separate from model training, so the model can be upgraded without replacing the format renderers.

The AI writer currently produces a bounded single-pass draft. Very long books, complex spreadsheets with formulas, charts, and multi-page brand systems still need iterative authoring and richer layout components. The SVG renderer is a simple vector poster, not a full illustration editor. Generated files are saved locally and served through Orbit's generated-file route.

### Multimodal workspace UI and private media

The Orbit web interface includes one chat plus visible upload and processing controls:
- **Vision:** upload an image and choose Understand image; the configured vision model analyses the actual image pixels.
- **Audio:** upload an audio file and choose Transcribe / understand audio; optional faster-whisper runs locally and returns detected language and transcript.
- **Video:** upload a video and choose Analyse video; ffmpeg samples up to six frames and the local vision/Whisper backends analyse available visual and audio content.
- **Creative Studio:** use the visible form to create PDF, DOCX, PPTX, XLSX, CSV, JSON, HTML, SVG, Markdown or TXT deliverables with selectable themes.

These are tool-backed capabilities, not capabilities learned natively by FirstGPT. Image understanding needs a configured vision model; audio/video need their optional local dependencies. Video analysis samples frames rather than understanding every frame continuously. Creative Studio uses FirstGPT to draft content and local exporters to create files.

**Private-media default:** `ORBIT_PRIVATE_MEDIA_ONLY=true` is the default. The vision adapter rejects non-loopback endpoints, including remote hosted vision APIs, while this setting is enabled. Audio transcription and video frame extraction run locally. Uploads, generated files, SQLite conversation history and owner preferences are stored on the device running Orbit; the app does not automatically publish or sync them to a public service. Orbit also binds to loopback by default and rejects a non-loopback host unless the explicit remote-access override is set.

This is local-by-default, not end-to-end encryption: files are not encrypted at rest by Orbit, other apps/processes with access to the device account may be able to read them, and enabling a remote vision endpoint or exposing the server changes the privacy boundary. Keep the server on loopback, use device security, and do not set `ORBIT_PRIVATE_MEDIA_ONLY=false` for sensitive uploads. Do not expose the prototype to a public network; it does not include user authentication.

### Images, files and image search

Uploads are limited to 20 MB per batch. Orbit previews common images (JPG/JPEG, PNG, WEBP, GIF, BMP, TIFF, AVIF and HEIC/HEIF) and can pass their actual pixels to a configured vision model. FirstGPT itself has not been trained with a vision encoder.

Common readable formats include:
- Text, Markdown, CSV, JSON, XML, HTML, RTF and source-code files.
- PDF, DOCX, XLSX, PPTX and OpenDocument files (ODT, ODS, ODP).
- EPUB books.
- ZIP, TAR, TGZ and TAR.GZ archives. Orbit inspects supported text entries without extracting or executing archive contents.
- Audio formats such as MP3, WAV, M4A, OGG and FLAC; video formats such as MP4, MOV, WEBM and MKV.

PDF text extraction uses pypdf; scanned-page OCR uses optional PDFium plus the configured vision model. Audio transcription uses optional faster-whisper. Video analysis samples up to six frames and, when ffmpeg and faster-whisper are available, transcribes up to five minutes of its audio track. This is sampled media understanding, not frame-by-frame temporal comprehension. Optional media dependencies are in requirements-media.txt; ffmpeg is installed separately.

Other file formats can still be safely stored and downloaded, but Orbit does not claim to understand a format without a parser. Legacy binary Office formats (DOC/XLS/PPT) and proprietary formats are not currently parsed.

Internet image search uses the Brave Search API. Set BRAVE_SEARCH_API_KEY. From the single chat, Orbit automatically uses strict SafeSearch by default and only relaxes filtering when the request clearly asks for mature results; the provider may still limit returned material. Search results appear inline with thumbnails and original source-page links. Respect source-site access controls and image licences.

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

The identity blueprint makes mature subjects part of ordinary conversation rather than a separate roleplay-only mode: direct adult vocabulary, relationships, intimacy, sexual health, mature humour, real-life questions and fiction are addressed naturally when relevant, without forcing every conversation in that direction. The vision adapter can analyse uploaded images when configured and now instructs the vision model to stick to observable details, avoid guessing identity/age/consent, and treat intimate media as sensitive. Uploads have a delete control, but local file storage alone does not make an exposed server private; use only on a trusted local connection and do not share the server publicly. The image backends download or connect to separate diffusion models; FirstGPT itself does not generate pixels. Output depends on the selected model and its capabilities.

## Scope and limitations

The GPT training foundation is complete as an engineering baseline, and the initial Orbit runtime foundations are now present. The notebook's real-data path now defaults to a larger WikiText-103 sample and a longer instruction-tuning run, but it still must be executed on the user's Colab GPU and its resulting checkpoint downloaded. This does not mean the model has learned useful language or that a complete autonomous assistant exists. Quality depends on corpus volume and quality, compute, model scale and subsequent training. The instruction-tuning pipeline and image-studio adapter/UI are implemented, but the included tuning examples are only a tiny demo. The optional open-corpus and Dolly instruction-data pipeline is implemented but must still be run on a GPU. The specialist team delegates research, image search, AST-based workspace inspection, bug tracing, code review, calculations, verification and attachment processing concurrently. It remains a bounded tool-using team sharing one model, not independent LLM instances. The autonomous coding workflow now applies exact, validated patches in dependency-ordered phases, assigns role-specialized architect, designer, frontend, backend, data, security and QA passes, reviews proposed changes, preserves backups and attempts bounded repairs. These role passes use separate task prompts and scoped file contexts but share the same model runtime and project workspace; they are not independent model instances or fully isolated sandboxes. Open-ended tool planning beyond registered tools and robust autonomous work across arbitrary repositories remain future engineering work. Public-site browser commands are available through the optional Playwright adapter, but they are explicit selector-based actions rather than autonomous account workflows. Native vision training inside FirstGPT, full audio/video comprehension, robust recovery across every external service, and extensive end-to-end testing also remain future work. Orbit's connected vision and optional media-processing adapters require their respective external/local backends. Orbit is currently a local prototype, not an always-on service.
