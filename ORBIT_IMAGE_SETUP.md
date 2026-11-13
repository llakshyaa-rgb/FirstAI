# Orbit image-generation setup

Orbit's FirstGPT language checkpoint does not generate pixels. Image creation is delegated to a separately installed ComfyUI server and an image model/workflow that you choose and have permission to use.

## What is implemented
- `orbit.images.ComfyUIImageBackend` submits ComfyUI API workflows, polls job history, and retrieves generated image bytes.
- `orbit.image_workflow.load_workflow_template` loads a ComfyUI API-format workflow JSON and replaces `{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, and `{{SEED}}` placeholders.
- The backend can run locally or on a separately managed machine reachable from the Orbit host.

## Configure
1. Install ComfyUI on a machine with suitable compute and install a compatible image checkpoint.
2. Export a workflow in ComfyUI's API format.
3. Use the included `workflows/comfyui_checkpoint_template.json` as a starting point. It contains placeholders for prompt, negative prompt, checkpoint filename, seed, width, height, steps and CFG; Orbit replaces these with typed values.
4. Keep the workflow file local and pass it to `load_workflow_template`.
5. Set `ORBIT_IMAGE_CHECKPOINT` to the exact filename shown in ComfyUI's `models/checkpoints` folder. Optional render settings are `ORBIT_IMAGE_WIDTH`, `ORBIT_IMAGE_HEIGHT`, `ORBIT_IMAGE_STEPS`, and `ORBIT_IMAGE_CFG`. The local server reads these when each image task starts.

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
The prompt can describe adult fictional characters, styling, pose, composition, and artistic anatomy. The actual image model and its configured restrictions determine what it will render. Do not assume that a model supports every degree of nudity or explicit sexual content. Never use the system for sexual imagery of minors or non-consensual intimate imagery. A public adult-facing service needs age checks, privacy protections, and appropriate safeguards.