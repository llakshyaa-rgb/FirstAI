"""Wan2.1 text-to-video backend with lazy, thread-safe loading."""
from __future__ import annotations
import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from ..base import MediaRequest
from ..registry import backend_registry
from .base import VideoBackend

OUTPUT_DIR = Path(os.environ.get("ORBIT_VIDEO_DIR", "/tmp/orbit/video"))
MODEL_ID = os.environ.get("ORBIT_WAN_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

@backend_registry.register
class WanBackend(VideoBackend):
    name = "wan"
    _pipe = None
    _pipe_lock = threading.Lock()

    def _load(self):
        if self._pipe is None:
            with self._pipe_lock:
                if self._pipe is None:
                    import torch  # type: ignore
                    from diffusers import WanPipeline  # type: ignore
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16 if device == "cuda" else torch.float32
                    pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
                    if device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.to(device)
                    self._pipe = pipe
        return self._pipe

    def _render(self, req: MediaRequest, fps: int, width: int, height: int,
                frame_count: int, context: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pipe = self._load()
        import torch  # type: ignore
        from diffusers.utils import export_to_video  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(int(req.seed or 0))
        negative = req.negative_prompt or "low quality, blurry, watermark, text, distorted"
        num_frames = max(16, frame_count)
        steps = int(req.extras.get("steps", 30))
        guidance = float(req.extras.get("guidance", 5.0))
        if steps < 1 or not 0 <= guidance <= 30:
            raise ValueError("steps must be positive and guidance must be between 0 and 30")
        result = pipe(prompt=req.prompt, negative_prompt=negative, height=height,
                      width=width, num_frames=num_frames, num_inference_steps=steps,
                      guidance_scale=guidance, generator=generator)
        digest = hashlib.sha256(
            f"{req.prompt}|{negative}|{req.seed}|wan|{width}x{height}|{num_frames}|{steps}|{guidance}".encode()
        ).hexdigest()[:16]
        path = OUTPUT_DIR / f"wan_{digest}.mp4"
        try:
            export_to_video(result.frames[0], str(path), fps=fps)
        except Exception:
            self._write_mp4_fallback(result.frames[0], path, fps)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Wan produced no video file")
        meta = {"model": MODEL_ID, "prompt": req.prompt, "negative_prompt": negative,
                "fps": fps, "width": width, "height": height, "num_frames": num_frames,
                "steps": steps, "guidance": guidance}
        return str(path), None, meta

    def _write_mp4_fallback(self, frames, path: Path, fps: int) -> None:
        try:
            import imageio.v2 as imageio  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"no video encoder available: {exc}") from exc
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
