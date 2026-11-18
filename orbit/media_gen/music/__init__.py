"""Music generation subpackage."""
from .base import MusicBackend, MusicResult
from .pipeline import MusicPipeline

# Register concrete backends as a side-effect of import.
from . import musicgen_backend, stable_audio_backend, vocal_backend  # noqa: F401

__all__ = ["MusicBackend", "MusicResult", "MusicPipeline"]
