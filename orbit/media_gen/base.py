"""Core dataclasses + abstract backend for all media kinds."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MediaKind(str, Enum):
    MUSIC = "music"
    VIDEO = "video"
    IMAGE = "image"


@dataclass
class MediaRequest:
    kind: MediaKind
    prompt: str
    duration_s: float = 10.0
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    negative_prompt: Optional[str] = None


@dataclass
class MediaResult:
    ok: bool
    request_id: str
    kind: MediaKind
    backend: str
    uri: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MediaBackend(ABC):
    """Every concrete backend implements `name`, `kind`, `generate`."""

    name: str = "unnamed"
    kind: MediaKind = MediaKind.IMAGE

    @abstractmethod
    def generate(
        self,
        req: MediaRequest,
        context: Optional[Dict[str, Any]] = None,
    ) -> MediaResult:
        ...

    def healthcheck(self) -> bool:
        return True


class BackendError(RuntimeError):
    """Raised by backends on unrecoverable failures."""
