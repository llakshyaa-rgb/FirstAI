"""Routes a MediaRequest to the correct registered backend."""
from __future__ import annotations

from typing import Optional

from .base import MediaBackend, MediaKind, MediaRequest, MediaResult
from .registry import Registry, backend_registry

# Default backend name per kind — override via env or config if needed.
_DEFAULTS = {
    MediaKind.MUSIC: "musicgen",
    MediaKind.VIDEO: "wan",
    MediaKind.IMAGE: "diffusers",
}


class Router:
    def __init__(self, registry: Optional[Registry] = None) -> None:
        self.registry = registry or backend_registry

    def resolve(self, kind: MediaKind, name: Optional[str] = None) -> MediaBackend:
        chosen = name or _DEFAULTS.get(kind)
        if chosen is None:
            raise KeyError(f"No default backend configured for kind={kind.value}")
        return self.registry.get(kind, chosen)

    def route(self, req: MediaRequest, backend_name: Optional[str] = None) -> MediaResult:
        try:
            backend = self.resolve(req.kind, backend_name)
        except KeyError as e:
            return MediaResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=backend_name or "?",
                error=str(e),
            )
        try:
            return backend.generate(req)
        except Exception as e:  # noqa: BLE001
            return MediaResult(
                ok=False,
                request_id=req.request_id,
                kind=req.kind,
                backend=backend.name,
                error=f"{type(e).__name__}: {e}",
            )


_default_router = Router()


def route_request(req: MediaRequest, backend_name: Optional[str] = None) -> MediaResult:
    return _default_router.route(req, backend_name)
