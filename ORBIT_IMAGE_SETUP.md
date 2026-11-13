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

## Uploads, image search and vision

Orbit's local browser UI also accepts uploads independently of image generation:
- JPG/JPEG, PNG, WEBP and GIF images (including mature images) are stored locally and previewed in the private local interface.
- PDF text extraction uses `pypdf`; DOCX, XLSX and PPTX text is extracted from their Open XML package; common text, code, JSON and CSV files are read as UTF-8.
- Any file extension can be uploaded and stored as an opaque file, but formats without a parser are not automatically decoded. ZIP, audio and video files are stored but not unpacked or transcribed.
- Files are size-limited to 20 MB each. Uploads are data, never executable code.

### Internet image search

Set a Brave Search API token in the environment before starting Orbit:

```bash
export BRAVE_SEARCH_API_KEY=your_brave_search_api_token
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

The Image Search panel uses Brave's image index. Its mature-results toggle switches between `safesearch=off` and `safesearch=strict`. The provider may return mature results when filtering is off, but results are not guaranteed, and the provider may still block illegal material. Orbit shows thumbnails and links to the original source page; it does not claim ownership or bypass a source site's access controls. Check the source's licence and terms before reusing images. The search API requires its own key and account; it is not bundled and may have usage limits or cost.

### Image understanding (optional)

FirstGPT itself is currently text-only. To let Orbit inspect an uploaded image, configure an Ollama-compatible vision endpoint and a vision-capable model:

```bash
export ORBIT_VISION_URL=http://127.0.0.1:11434/api/generate
export ORBIT_VISION_MODEL=llava
```

Restart Orbit, upload an image, and choose **Ask Orbit to analyse**. The endpoint/model must support image input. The vision adapter sends the image to that configured endpoint; if the endpoint is remote, the image leaves the device and is handled by that provider. Do not send private images to an endpoint you do not trust. A separate vision model is not FirstGPT's own learned vision ability, and mature-image interpretation depends on that model's own capabilities and restrictions.
