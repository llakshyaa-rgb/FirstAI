"""Thread-safe backend registry with lazy singleton instances."""
from __future__ import annotations
import threading
from typing import Dict, List, Optional, Type
from .base import MediaBackend, MediaKind

class Registry:
    def __init__(self) -> None:
        self._classes: Dict[MediaKind, Dict[str, Type[MediaBackend]]] = {}
        self._instances: Dict[str, MediaBackend] = {}
        self._lock = threading.RLock()

    def register(self, backend_cls: Type[MediaBackend]) -> Type[MediaBackend]:
        kind, name = backend_cls.kind, backend_cls.name
        if not isinstance(kind, MediaKind) or not name or name == "unnamed":
            raise ValueError("backend must define a MediaKind and non-empty name")
        with self._lock:
            self._classes.setdefault(kind, {})[name] = backend_cls
            self._instances.pop(f"{kind.value}:{name}", None)
        return backend_cls

    def get(self, kind: MediaKind, name: str) -> MediaBackend:
        if not isinstance(kind, MediaKind):
            raise TypeError("kind must be a MediaKind")
        key = f"{kind.value}:{name}"
        with self._lock:
            if key in self._instances:
                return self._instances[key]
            try:
                cls = self._classes[kind][name]
            except KeyError as exc:
                raise KeyError(f"No backend registered for {key}") from exc
            instance = cls()
            self._instances[key] = instance
            return instance

    def list(self, kind: Optional[MediaKind] = None) -> List[str]:
        with self._lock:
            if kind is None:
                return [f"{k.value}:{n}" for k, group in self._classes.items() for n in group]
            return list(self._classes.get(kind, {}).keys())

backend_registry = Registry()
