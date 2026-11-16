# Orbit video generation

Orbit has a dedicated Video Studio and a text-to-video route. It submits a ComfyUI API-format workflow to a GPU host, polls the job, downloads returned video files, and displays playable MP4/WEBM/MOV/GIF/WEBP results in chat.

## Requirements

- ComfyUI on a trusted GPU host reachable by Orbit.
- A compatible video model and required custom nodes. The example assumes a compatible latent-video workflow and VideoHelperSuite's `VHS_VideoCombine`; adapt node names and inputs to your selected model.
- An API-format workflow exported from ComfyUI. The included JSON is a structural example, not a universal ready-to-run workflow; verify node classes and model compatibility against your installation.

## Configure

```bash
export ORBIT_COMFYUI_URL=http://127.0.0.1:8188
export ORBIT_VIDEO_WORKFLOW="$PWD/examples/comfyui/orbit_video_workflow_api.example.json"
export ORBIT_VIDEO_WIDTH=576
export ORBIT_VIDEO_HEIGHT=576
export ORBIT_VIDEO_FRAMES=49
export ORBIT_VIDEO_FPS=12
export ORBIT_VIDEO_STEPS=20
export ORBIT_VIDEO_CFG=6
export ORBIT_VIDEO_TIMEOUT=1800
python -m orbit.server --checkpoint checkpoints/orbit-instruct.pt --port 8767
```

Send a video request in chat or open **Video Studio**. Supported workflow placeholders: `{{PROMPT}}`, `{{NEGATIVE_PROMPT}}`, `{{SEED}}`, `{{WIDTH}}`, `{{HEIGHT}}`, `{{FRAMES}}`, `{{FPS}}`, `{{STEPS}}`, and `{{CFG}}`.

## Prompt scope and privacy

Orbit's media prompt guard rejects explicit nudity and sexual content. Supported directions include editorial fashion, swimwear and lingerie design, non-explicit classical figure studies, character animation, cinematic scenes, and tasteful portraiture.

Keep ComfyUI on loopback or a private trusted network. Do not expose its unauthenticated API to the public internet or commit credentials. The Android phone can host Orbit's interface, but video rendering requires a suitable GPU host; virtual RAM does not replace GPU memory.

## Troubleshooting

- If the workflow returns no video, ensure its final node emits a ComfyUI `videos` or `gifs` output in the history response.
- If a node class is missing, install the custom node package required by your selected model and restart ComfyUI.
- If VRAM runs out, reduce resolution, frame count, batch size or model size.
- A configured workflow is not proof of a successful render; Orbit reports actual backend errors and only shows media after it downloads the output.
