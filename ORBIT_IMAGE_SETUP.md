# Orbit image-generation setup

Orbit's FirstGPT language checkpoint does not generate pixels. Image creation is delegated to a separately installed ComfyUI server and an image model/workflow that you choose and have permission to use.

## What is implemented
- `orbit.images.ComfyUIImageBackend` submits ComfyUI API workflows, polls job history, and retrieves generated image bytes.
- `orbit.image_workflow.load_workflow_template` loads a ComfyUI API-format workflow JSON and replaces prompt, negative prompt, checkpoint, seed, dimensions, steps and CFG placeholders with validated typed values.
- The backend can run locally or on a separately managed machine reachable from the Orbit host.

## Configure
1. Install ComfyUI on a machine with suitable compute and install a compatible image checkpoint.
2. Export a workflow in ComfyUI's API format.
3. Use the included `workflows/comfyui_checkpoint_template.json` as a starting point. It contains placeholders for prompt, negative prompt, checkpoint filename, seed, width, height, steps and CFG; Orbit replaces these with typed values.
4. Keep the workflow file local and pass it to `load_workflow_template`.
5. Set `ORBIT_IMAGE_CHECKPOINT` to the exact filename shown in ComfyUI's `models/checkpoints` folder. Optional render settings are `ORBIT_IMAGE_WIDTH`, `ORBIT_IMAGE_HEIGHT`, `ORBIT_IMAGE_STEPS`, and `ORBIT_IMAGE_CFG`. You may also set `ORBIT_COMFYUI_FALLBACK_URL` to a second compatible ComfyUI endpoint; Orbit announces a primary failure and retries the image job there. The local server reads these when each image task starts.

Example:

```python
from orbit.images import ComfyUIImageBackend
from orbit.image_workflow import load_workflow_template

workflow = load_workflow_template(
    "workflows/comfyui_checkpoint_template.json",
    prompt="A detailed character concept illustration, cinematic lighting",
    negative_prompt="low quality, distorted anatomy",
    seed=12345,
    checkpoint="your-installed-model.safetensors",
    width=768, height=768, steps=24, cfg=7.0,
 )
images = ComfyUIImageBackend().generate(workflow)
for index, item in enumerate(images):
    with open(f"orbit-image-{index}.png", "wb") as output:
        output.write(item["content"])
```

The workflow must be exported in API format and its node inputs must match the installed ComfyUI graph. Orbit cannot infer a compatible graph from a prompt alone.

## Mature artwork
The prompt can describe adult fictional characters, styling, pose, composition, and artistic anatomy. The actual image model and its configured restrictions determine what it will render. Do not assume that a model supports every degree of nudity or explicit sexual content. Never use the system for sexual imagery of minors or non-consensual intimate imagery.

## Uploads, image search and multimodal understanding

Orbit accepts image uploads (including mature subject matter), previews them, and can attach up to five uploaded files to an ordinary chat message. Use **Attach to next chat** on an upload, then ask a question in the normal chat composer. Orbit processes the attached material before generating its answer.

### Image understanding

FirstGPT's current decoder-only checkpoint is text-only; image reading is provided by a vision-capable model through an adapter. The adapter supports Ollama and OpenAI-compatible vision servers (including compatible local servers such as llama.cpp builds with multimodal support).

Ollama example:

\`\`\`bash
export ORBIT_VISION_PROVIDER=ollama
export ORBIT_VISION_URL=http://127.0.0.1:11434/api/generate
export ORBIT_VISION_MODEL=llava
\`\`\`

OpenAI-compatible server example:

\`\`\`bash
export ORBIT_VISION_PROVIDER=openai-compatible
export ORBIT_VISION_URL=http://127.0.0.1:8080
export ORBIT_VISION_MODEL=your-vision-model
export ORBIT_VISION_API_KEY=your-local-or-remote-key
\`\`\`

Restart Orbit after setting the variables. Upload an image, choose **Attach to next chat**, and ask what is in it, ask it to read visible text, compare details, or describe the scene. The direct **Ask Orbit to analyse** button is also available. The configured vision model receives the actual image bytes/pixels; Orbit passes its description into FirstGPT as context. If the vision endpoint is remote, the image is sent to that provider. Do not send private images to an endpoint you do not trust.

This is real image reading in the Orbit application, but it is not yet a vision capability learned by FirstGPT itself. A native FirstGPT vision encoder and image-text training stage remain future model research; merely adding an adapter does not train that capability into the language checkpoint.

### Other file and media processing

- Text, CSV, JSON, code and common office documents are extracted and can be attached to chat.
- PDF text is extracted with pypdf. For scanned/image-only PDFs, optional PDFium rendering plus the configured vision model provides OCR on up to 12 pages.
- ZIP archives are inspected without extracting files to disk; supported text files inside are read with limits.
- Audio transcription uses optional faster-whisper with a small model by default. It can run on CPU, but performance and memory depend on the device/model.
- Video processing uses ffmpeg to sample up to six frames and sends those frames to the configured vision model. This analyses sampled frames, not every frame or the full audio track.
- Unsupported formats are safely stored as opaque files and are not falsely described as understood.

Install optional media dependencies only when needed:

\`\`\`bash
pip install -r requirements-media.txt
\`\`\`

Video processing also requires ffmpeg installed and available on PATH. Optional audio settings: ORBIT_WHISPER_MODEL=tiny, ORBIT_WHISPER_DEVICE=cpu, and ORBIT_WHISPER_COMPUTE_TYPE=int8. Large models can be slow or unsuitable on a phone.

### Internet image search

Set a Brave Search API token before starting Orbit:

\`\`\`bash
export BRAVE_SEARCH_API_KEY=your_brave_search_api_token
\`\`\`

The Image Search panel uses Brave's image index. Its mature-results toggle switches between safesearch=off and safesearch=strict. The provider may return mature results when filtering is off, but results are not guaranteed and provider restrictions still apply. Orbit shows thumbnails and links to source pages; it does not bypass source-site access controls. Check licensing and reuse terms before reposting an image. Brave API access may have account requirements, rate limits or costs.
