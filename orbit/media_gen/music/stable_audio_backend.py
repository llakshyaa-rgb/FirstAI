"""Stable Audio Open backend (Stability AI). Lazy-loaded diffusers pipeline."""
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
class StableAudioBackend(MusicBackend):
    name = "stable_audio"
    _pipe = None
    _pipe_lock = threading.Lock()

    def _load(self):
        if self._pipe is None:
            with self._pipe_lock:
                if self._pipe is None:
                    import torch  # type: ignore
                    from diffusers import StableAudioPipeline  # type: ignore
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._pipe = StableAudioPipeline.from_pretrained(
                        "stabilityai/stable-audio-open-1.0",
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    ).to(device)
        return self._pipe

    def _synthesize(
        self,
        req: MediaRequest,
        sample_rate: int,
        channels: int,
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pipe = self._load()

        import torch  # type: ignore
        from scipy.io.wavfile import write as wav_write  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device).manual_seed(int(req.seed or 0))

        audio = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "low quality, distorted",
            num_inference_steps=int(req.extras.get("steps", 100)),
            audio_end_in_s=float(req.duration_s),
            num_waveforms_per_prompt=1,
            generator=generator,
        ).audios

        output = audio[0].T.float().cpu().numpy()
        digest = hashlib.sha256(
            f"{req.prompt}|{req.negative_prompt}|{req.seed}|{req.duration_s}|{sample_rate}|{channels}|stable_audio".encode()
        ).hexdigest()[:16]
        path = OUTPUT_DIR / f"stable_audio_{digest}.wav"

        out_sr = int(pipe.vae.sampling_rate)
        if channels == 1 and output.ndim == 2 and output.shape[1] > 1:
            output = output.mean(axis=1, keepdims=True)
        elif channels == 2 and output.ndim == 2 and output.shape[1] == 1:
            output = output.repeat(2, axis=1)
        if sample_rate != out_sr:
            from math import gcd
            from scipy.signal import resample_poly  # type: ignore
            divisor = gcd(sample_rate, out_sr)
            output = resample_poly(output, sample_rate // divisor,
                                   out_sr // divisor, axis=0)
            out_sr = sample_rate
        wav_write(str(path), out_sr, output)

        meta: Dict[str, Any] = {
            "model": "stabilityai/stable-audio-open-1.0",
            "duration_s": req.duration_s,
            "prompt": req.prompt,
            "sample_rate": out_sr,
            "bpm": req.extras.get("bpm"),
        }
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Stable Audio produced no audio file")
        return str(path), meta
