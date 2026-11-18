"""MusicGen backend (Meta AudioCraft). Lazy-imports torch/audiocraft on first use."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from ..base import MediaRequest
from ..registry import backend_registry
from .base import MusicBackend

OUTPUT_DIR = Path(os.environ.get("ORBIT_MUSIC_DIR", "/tmp/orbit/music"))


@backend_registry.register
class MusicGenBackend(MusicBackend):
    name = "musicgen"
    _model = None

    def _load(self):
        if self._model is None:
            from audiocraft.models import MusicGen  # type: ignore
            self._model = MusicGen.get_pretrained("facebook/musicgen-small")
        return self._model

    def _synthesize(
        self,
        req: MediaRequest,
        sample_rate: int,
        channels: int,
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model = self._load()
        model.set_generation_params(duration=req.duration_s)

        wav = model.generate([req.prompt])
        digest = hashlib.sha1(f"{req.prompt}|{req.seed}".encode()).hexdigest()[:12]
        path = OUTPUT_DIR / f"musicgen_{digest}.wav"

        try:
            from audiocraft.data.audio import audio_write  # type: ignore
            audio_write(
                str(path.with_suffix("")),
                wav[0].cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )
        except Exception:  # noqa: BLE001
            # Fallback: torchaudio direct write
            import torchaudio  # type: ignore
            torchaudio.save(str(path), wav[0].cpu(), model.sample_rate)

        meta: Dict[str, Any] = {
            "model": "facebook/musicgen-small",
            "duration_s": req.duration_s,
            "prompt": req.prompt,
            "sample_rate": model.sample_rate,
            "bpm": req.extras.get("bpm"),
        }
        return str(path), meta
