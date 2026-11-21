"""MusicGen backend (Meta AudioCraft). Lazy-imports torch/audiocraft on first use."""
from __future__ import annotations

import hashlib
import os
import threading
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
    _model_lock = threading.Lock()
    _generation_lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._model_lock:
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
        # AudioCraft's generation settings and torch RNG are process-global.
        with self._generation_lock:
            model.set_generation_params(duration=req.duration_s)
            if req.seed is not None:
                import torch  # type: ignore
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
                    torch.manual_seed(int(req.seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(req.seed))
                    wav = model.generate([req.prompt])
            else:
                wav = model.generate([req.prompt])
        digest = hashlib.sha256(
            f"{req.prompt}|{req.seed}|{req.duration_s}|{sample_rate}|{channels}".encode()
        ).hexdigest()[:16]
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
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("MusicGen produced no audio file")
        # AudioCraft writes at the model's native rate; convert to the
        # caller-requested PCM format instead of merely reporting it.
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if sample_rate != int(model.sample_rate) or channels != 2:
            if not ffmpeg:
                raise RuntimeError("ffmpeg is required to convert MusicGen output format")
            converted = path.with_name(path.stem + ".converted.wav")
            proc = subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-ar", str(sample_rate),
                 "-ac", str(channels), str(converted)],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0 or not converted.is_file() or converted.stat().st_size == 0:
                raise RuntimeError(f"MusicGen audio conversion failed: {proc.stderr[-1200:]}")
            converted.replace(path)
        meta["sample_rate"] = sample_rate
        meta["channels"] = channels
        return str(path), meta
