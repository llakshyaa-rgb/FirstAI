"""Backend registry — maps (kind, name) -> backend class, lazily instantiated."""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import MediaBackend, MediaKind


class Registry:
    def __init__(self) -> None:
        self._classes: Dict[MediaKind, Dict[str, Type[MediaBackend]]] = {}
        self._instances: Dict[str, MediaBackend] = {}

    def register(self, backend_cls: Type[MediaBackend]) -> Type[MediaBackend]:
        kind = backend_cls.kind
        self._classes.setdefault(kind, {})[backend_cls.name] = backend_cls
        return backend_cls

    def get(self, kind: MediaKind, name: str) -> MediaBackend:
        key = f"{kind.value}:{name}"
        if key in self._instances:
            return self._instances[key]
        try:
            cls = self._classes[kind][name]
        except KeyError as e:
            raise KeyError(f"No backend registered for {key}") from e
        inst = cls()
        self._instances[key] = inst
        return inst

    def list(self, kind: Optional[MediaKind] = None) -> List[str]:
        if kind is None:
            return [f"{k.value}:{n}" for k, m in self._classes.items() for n in m]
        return list(self._classes.get(kind, {}).keys())


backend_registry = Registry()
