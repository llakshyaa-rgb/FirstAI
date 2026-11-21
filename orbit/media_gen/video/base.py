"""Video backend abstraction with request validation."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from ..base import MediaBackend, MediaKind, MediaRequest, MediaResult

@dataclass
class VideoResult(MediaResult):
    fps: int = 24
    width: int = 1024
    height: int = 576
    frame_count: int = 0
    audio_uri: Optional[str] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)

class VideoBackend(MediaBackend):
    kind = MediaKind.VIDEO
    default_fps = 24
    default_width = 1024
    default_height = 576

    def generate(self, req: MediaRequest, context: Optional[Dict[str, Any]] = None) -> VideoResult:
        def fail(message: str) -> VideoResult:
            return VideoResult(False, req.request_id, req.kind, self.name, error=message)
        if req.kind != MediaKind.VIDEO:
            return fail(f"{self.name} only handles VIDEO, got {req.kind.value}")
        try:
            if not isinstance(req.prompt, str) or not req.prompt.strip():
                raise ValueError("prompt cannot be empty")
            duration = float(req.duration_s)
            if not math.isfinite(duration) or not 0 < duration <= 600:
                raise ValueError("duration_s must be finite and between 0 and 600 seconds")
            fps = int(req.extras.get("fps", self.default_fps))
            width = int(req.extras.get("width", self.default_width))
            height = int(req.extras.get("height", self.default_height))
            if not 1 <= fps <= 120:
                raise ValueError("fps must be between 1 and 120")
            if not 64 <= width <= 4096 or not 64 <= height <= 4096:
                raise ValueError("dimensions must be between 64 and 4096")
            frames = max(1, int(round(fps * duration)))
            uri, audio_uri, meta = self._render(req, fps, width, height, frames, context or {})
            if not uri or not Path(uri).is_file():
                raise RuntimeError("backend did not create its output video")
            meta = meta or {}
        except Exception as exc:
            return fail(f"{type(exc).__name__}: {exc}")
        return VideoResult(True, req.request_id, req.kind, self.name, uri=uri, meta=meta,
                           fps=fps, width=width, height=height, frame_count=frames,
                           audio_uri=audio_uri,
                           extra_meta={k:v for k,v in meta.items() if k not in {"fps","width","height"}})

    def _render(self, req: MediaRequest, fps: int, width: int, height: int,
                frame_count: int, context: Dict[str, Any]):
        raise NotImplementedError
