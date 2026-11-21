"""Routes media requests to registered backends and validates responses."""
from __future__ import annotations
from typing import Optional
from .base import MediaBackend, MediaKind, MediaRequest, MediaResult
from .registry import Registry, backend_registry

_DEFAULTS = {MediaKind.MUSIC: "musicgen", MediaKind.VIDEO: "wan", MediaKind.IMAGE: "diffusers"}

class Router:
    def __init__(self, registry: Optional[Registry] = None) -> None:
        self.registry = registry if registry is not None else backend_registry

    def resolve(self, kind: MediaKind, name: Optional[str] = None) -> MediaBackend:
        chosen = name or _DEFAULTS.get(kind)
        if chosen is None:
            raise KeyError(f"No default backend configured for kind={kind.value}")
        return self.registry.get(kind, chosen)

    def route(self, req: MediaRequest, backend_name: Optional[str] = None) -> MediaResult:
        if not isinstance(req, MediaRequest):
            raise TypeError("req must be a MediaRequest")
        try:
            backend = self.resolve(req.kind, backend_name)
        except KeyError as exc:
            return MediaResult(False, req.request_id, req.kind, backend_name or "?", error=str(exc))
        try:
            result = backend.generate(req)
            if not isinstance(result, MediaResult):
                raise TypeError(f"{backend.name}.generate() did not return MediaResult")
            if result.request_id != req.request_id or result.kind != req.kind:
                raise ValueError("backend returned a result for a different request")
            if result.ok and not result.uri:
                raise ValueError("backend reported success without an output URI")
            return result
        except Exception as exc:
            return MediaResult(False, req.request_id, req.kind, backend.name,
                               error=f"{type(exc).__name__}: {exc}")

_default_router = Router()
def route_request(req: MediaRequest, backend_name: Optional[str] = None) -> MediaResult:
    return _default_router.route(req, backend_name)
