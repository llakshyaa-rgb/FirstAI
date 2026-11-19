"""Video-specific backend abstraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    """Base for all video backends. Subclasses implement `_render`."""

    kind: MediaKind = MediaKind.VIDEO
    default_fps: int = 24
    default_width: int = 1024
    default_height: int = 576

    def generate(
        self,
        req: MediaRequest,
        context: Optional[Dict[str, Any]] = None,
    ) -> VideoResult:
        if req.kind != MediaKind.VIDEO:
            return VideoResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=self.name,
                error=f"{self.name} only handles VIDEO, got {req.kind.value}",
            )

        fps = int(req.extras.get("fps", self.default_fps))
        width = int(req.extras.get("width", self.default_width))
        height = int(req.extras.get("height", self.default_height))
        frame_count = int(round(fps * req.duration_s))

        try:
            uri, audio_uri, meta = self._render(
                req, fps, width, height, frame_count, context or {}
            )
        except Exception as e:  # noqa: BLE001
            return VideoResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=self.name,
                error=f"{type(e).__name__}: {e}",
            )

        return VideoResult(
            ok=True,
            request_id=req.request_id,
            kind=req.kind,
            backend=self.name,
            uri=uri,
            meta=meta,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            audio_uri=audio_uri,
            extra_meta={k: v for k, v in meta.items()
                        if k not in {"fps", "width", "height"}},
        )

    def _render(
        self,
        req: MediaRequest,
        fps: int,
        width: int,
        height: int,
        frame_count: int,
        context: Dict[str, Any],
    ):
        """Return (video_uri, audio_uri_or_None, meta). Override in subclasses."""
        raise NotImplementedError
