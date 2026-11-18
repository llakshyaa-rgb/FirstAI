"""Music-specific backend abstraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import MediaBackend, MediaKind, MediaRequest, MediaResult


@dataclass
class MusicResult(MediaResult):
    bpm: Optional[float] = None
    sample_rate: int = 44100
    channels: int = 2
    extra_meta: Dict[str, Any] = field(default_factory=dict)


class MusicBackend(MediaBackend):
    """Base for all music backends. Subclasses implement `_synthesize`."""

    kind: MediaKind = MediaKind.MUSIC
    default_sample_rate: int = 44100
    default_channels: int = 2

    def generate(
        self,
        req: MediaRequest,
        context: Optional[Dict[str, Any]] = None,
    ) -> MusicResult:
        if req.kind != MediaKind.MUSIC:
            return MusicResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=self.name,
                error=f"{self.name} only handles MUSIC, got {req.kind.value}",
            )

        sample_rate = int(req.extras.get("sample_rate", self.default_sample_rate))
        channels = int(req.extras.get("channels", self.default_channels))

        try:
            uri, meta = self._synthesize(req, sample_rate, channels, context or {})
        except Exception as e:  # noqa: BLE001
            return MusicResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=self.name,
                error=f"{type(e).__name__}: {e}",
            )

        return MusicResult(
            ok=True,
            request_id=req.request_id,
            kind=req.kind,
            backend=self.name,
            uri=uri,
            meta=meta,
            bpm=meta.get("bpm"),
            sample_rate=sample_rate,
            channels=channels,
            extra_meta={k: v for k, v in meta.items() if k not in {"bpm"}},
        )

    def _synthesize(
        self,
        req: MediaRequest,
        sample_rate: int,
        channels: int,
        context: Dict[str, Any],
    ):
        """Return (uri, meta). Override in subclasses."""
        raise NotImplementedError
