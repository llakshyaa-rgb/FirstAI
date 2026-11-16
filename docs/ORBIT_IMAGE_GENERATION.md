# Orbit image generation setup

Orbit already routes image-generation requests from the normal chat composer. The language model does not generate pixels itself: Orbit sends the prompt to a separately hosted image backend and displays the returned image in the conversation.

## Option A: ComfyUI (recommended when you have a separate GPU machine)

1. Install ComfyUI on the machine that will render images.
2. Start ComfyUI with its API enabled (the standard server exposes `/system_stats`, `/prompt`, `/history` and `/view`).
3. In ComfyUI, load a checkpoint you are licensed to use and export an **API-format** workflow.
4. Use `examples/comfyui/orbit_workflow_api.example.json` as a starting workflow. It uses Orbit's placeholders: `{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{CHECKPOINT}}`, `{{SEED}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{STEPS}}`, and `{{CFG}}`.
5. Set the environment variables in the same shell that launches Orbit:

```bash
export ORBIT_IMAGE_BACKEND=comfyui
export ORBIT_COMFYUI_URL=http://127.0.0.1:8188
export ORBIT_IMAGE_WORKFLOW="$PWD/examples/comfyui/orbit_workflow_api.example.json"
export ORBIT_IMAGE_CHECKPOINT="your-checkpoint-file.safetensors"
export ORBIT_IMAGE_WIDTH=768
export ORBIT_IMAGE_HEIGHT=768
export ORBIT_IMAGE_STEPS=24
export ORBIT_IMAGE_CFG=7
python -m orbit.doctor
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

If ComfyUI runs on another machine, replace the loopback URL with a private, trusted network address reachable from the Orbit host. Do not expose an unauthenticated ComfyUI server to the public internet. The optional `ORBIT_COMFYUI_FALLBACK_URL` is a second ComfyUI endpoint used only if the primary endpoint fails.

## Option B: Diffusers in the Orbit Python environment

This is convenient for a GPU workstation or hosted GPU runtime, but it downloads model weights and can use substantial disk space and VRAM.

```bash
pip install -r requirements-image.txt
export ORBIT_IMAGE_BACKEND=diffusers
export ORBIT_IMAGE_MODEL="Lykon/dreamshaper-8"
export ORBIT_IMAGE_DEVICE=auto
export ORBIT_IMAGE_WIDTH=512
export ORBIT_IMAGE_HEIGHT=512
export ORBIT_IMAGE_STEPS=20
export ORBIT_IMAGE_CFG=7
python -m orbit.doctor
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

Choose a model whose licence and usage terms fit your project. The default model ID is only a configurable starting point, not a guarantee that it is available, suitable for every task, or compatible with every device. CPU generation is generally very slow; Android/Termux is not a recommended Diffusers host.

## How to use it

Open Orbit's local web UI and send a normal request such as:

`Generate an editorial fashion illustration of an adult model wearing a tailored evening outfit, studio lighting, full-length composition.`

Orbit queues the render, reports progress, stores the resulting file under its generated-media folder, and posts the image in the chat. If the image backend is not configured or reachable, Orbit reports the actual setup error; it does not pretend an image was created.

## Provider/API integrations

A third-party website can only be integrated as a backend when it publishes a documented API that permits programmatic image generation and returns an image or a retrievable job result. A website's interactive chat page is not itself an API. Do not put account passwords, browser cookies, or private access tokens in source code or commit them to Git. Add credentials through environment variables or a local, untracked environment file only after verifying the provider's official API documentation and terms.

## Device notes

- The Android phone can host Orbit's lightweight chat UI and CPU language model, but it is not a practical host for large diffusion image models.
- Use a separate GPU machine or hosted GPU endpoint for image rendering.
- Image-generation model weights are separate from FirstGPT's text checkpoint. Adding an image backend does not train FirstGPT or make it a vision model.
