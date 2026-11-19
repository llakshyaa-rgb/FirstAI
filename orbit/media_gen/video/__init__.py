"""Video generation subpackage."""
from .base import VideoBackend, VideoResult
from .postprocess import interpolate, upscale, add_audio, concat

# Register concrete backends as a side-effect of import.
from . import wan_backend, ltx_backend  # noqa: F401

__all__ = [
    "VideoBackend",
    "VideoResult",
    "interpolate",
    "upscale",
    "add_audio",
    "concat",
]
