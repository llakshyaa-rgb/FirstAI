"""Stable Audio Open backend (Stability AI). Lazy-loaded diffusers pipeline."""
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
class StableAudioBackend(MusicBackend):
    name = "stable_audio"
    _pipe = None

    def _load(self):
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
        digest = hashlib.sha1(
            f"{req.prompt}|{req.seed}|stable_audio".encode()
        ).hexdigest()[:12]
        path = OUTPUT_DIR / f"stable_audio_{digest}.wav"

        out_sr = pipe.vae.sampling_rate
        if channels == 1 and output.ndim == 2 and output.shape[1] > 1:
            output = output.mean(axis=1, keepdims=True)
        wav_write(str(path), out_sr, output)

        meta: Dict[str, Any] = {
            "model": "stabilityai/stable-audio-open-1.0",
            "duration_s": req.duration_s,
            "prompt": req.prompt,
            "sample_rate": out_sr,
            "bpm": req.extras.get("bpm"),
        }
        return str(path), meta
