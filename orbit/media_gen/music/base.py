"""Music-specific backend abstraction with validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from ..base import MediaBackend, MediaKind, MediaRequest, MediaResult

@dataclass
class MusicResult(MediaResult):
    bpm: Optional[float] = None
    sample_rate: int = 44100
    channels: int = 2
    extra_meta: Dict[str, Any] = field(default_factory=dict)

class MusicBackend(MediaBackend):
    kind = MediaKind.MUSIC
    default_sample_rate = 44100
    default_channels = 2

    def generate(self, req: MediaRequest, context: Optional[Dict[str, Any]] = None) -> MusicResult:
        def fail(message: str) -> MusicResult:
            return MusicResult(False, req.request_id, req.kind, self.name, error=message)
        if req.kind != MediaKind.MUSIC:
            return fail(f"{self.name} only handles MUSIC, got {req.kind.value}")
        try:
            if not isinstance(req.prompt, str) or not req.prompt.strip():
                raise ValueError("prompt cannot be empty")
            if req.duration_s <= 0:
                raise ValueError("duration_s must be positive")
            sample_rate = int(req.extras.get("sample_rate", self.default_sample_rate))
            channels = int(req.extras.get("channels", self.default_channels))
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("sample_rate must be between 8000 and 192000")
            if channels not in (1, 2):
                raise ValueError("channels must be 1 or 2")
            uri, meta = self._synthesize(req, sample_rate, channels, context or {})
            if not uri or not Path(uri).is_file():
                raise RuntimeError("backend did not create its output file")
            meta = meta or {}
        except Exception as exc:
            return fail(f"{type(exc).__name__}: {exc}")
        return MusicResult(True, req.request_id, req.kind, self.name, uri=uri, meta=meta,
                           bpm=meta.get("bpm"), sample_rate=sample_rate, channels=channels,
                           extra_meta={k:v for k,v in meta.items() if k != "bpm"})

    def _synthesize(self, req: MediaRequest, sample_rate: int, channels: int,
                    context: Dict[str, Any]):
        raise NotImplementedError
