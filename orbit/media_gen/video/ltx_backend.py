"""LTX-Video (Lightricks) backend. Lazy-loads diffusers LTXPipeline."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..base import MediaRequest
from ..registry import backend_registry
from .base import VideoBackend

OUTPUT_DIR = Path(os.environ.get("ORBIT_VIDEO_DIR", "/tmp/orbit/video"))
MODEL_ID = os.environ.get(
    "ORBIT_LTX_MODEL", "Lightricks/LTX-Video"
)


@backend_registry.register
class LTXBackend(VideoBackend):
    name = "ltx"
    default_fps = 24
    default_width = 768
    default_height = 512

    _pipe = None

    def _load(self):
        if self._pipe is None:
            import torch  # type: ignore
            from diffusers import LTXPipeline  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self._pipe = LTXPipeline.from_pretrained(
                MODEL_ID, torch_dtype=dtype
            ).to(device)
            if device == "cuda":
                try:
                    self._pipe.enable_model_cpu_offload()
                except Exception:  # noqa: 001
                    pass
                try:
                    self._pipe.vae.enable_tiling()
                except Exception:  # noqa: 001
                    pass
        return self._pipe

    def _render(
        self,
        req: MediaRequest,
        fps: int,
        width: int,
        height: int,
        frame_count: int,
        context: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pipe = self._load()

        import torch  # type: ignore
        from diffusers.utils import export_to_video  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(int(req.seed or 0))

        # LTX likes dimensions divisible by 32 and frame count = 8k+1.
        num_frames = max(9, (frame_count // 8) * 8 + 1)
        steps = int(req.extras.get("steps", 40))
        guidance = float(req.extras.get("guidance", 3.0))
        negative = req.negative_prompt or (
            "worst quality, inconsistent motion, blurry, jittery, distorted"
        )

        result = pipe(
            prompt=req.prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )

        digest = hashlib.sha1(
            f"{req.prompt}|{req.seed}|ltx|{width}x{height}|{num_frames}".encode()
        ).hexdigest()[:12]
        path = OUTPUT_DIR / f"ltx_{digest}.mp4"

        try:
            export_to_video(result.frames[0], str(path), fps=fps)
        except Exception:  # noqa: 001
            self._write_mp4_fallback(result.frames[0], path, fps)

        meta: Dict[str, Any] = {
            "model": MODEL_ID,
            "prompt": req.prompt,
            "negative_prompt": negative,
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": steps,
            "guidance": guidance,
        }
        return str(path), None, meta

    def _write_mp4_fallback(self, frames, path: Path, fps: int) -> None:
        try:
            import imageio.v2 as imageio  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:  # noqa: 001
            raise RuntimeError(f"no video encoder available: {e}") from e

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
