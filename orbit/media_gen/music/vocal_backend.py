"""Vocal synthesis backend wrapping an existing TTS/voice pipeline.

Renders sung/spoken vocal layers and optionally mixes them under an
instrumental produced by another backend.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base import MediaRequest
from ..registry import backend_registry
from .base import MusicBackend

OUTPUT_DIR = Path(os.environ.get("ORBIT_MUSIC_DIR", "/tmp/orbit/music"))


@backend_registry.register
class VocalBackend(MusicBackend):
    name = "vocal"
    default_sample_rate = 44100
    default_channels = 1

    def _synthesize(
        self,
        req: MediaRequest,
        sample_rate: int,
        channels: int,
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        voice: str = str(req.extras.get("voice", "default"))
        lyrics: Optional[str] = req.extras.get("lyrics")
        instrumental_uri: Optional[str] = req.extras.get("instrumental_uri")
        pitch_shift: float = float(req.extras.get("pitch_semitones", 0.0))

        text = lyrics or req.prompt
        digest = hashlib.sha1(
            f"{text}|{voice}|{req.seed}".encode()
        ).hexdigest()[:12]
        raw_path = OUTPUT_DIR / f"vocal_raw_{digest}.wav"

        self._render_tts(text, voice, raw_path, sample_rate)

        if pitch_shift:
            self._pitch_shift(raw_path, pitch_shift, sample_rate)

        if instrumental_uri:
            final_path = OUTPUT_DIR / f"vocal_mix_{digest}.wav"
            self._mix(instrumental_uri, str(raw_path), str(final_path))
        else:
            final_path = raw_path

        if channels == 1:
            self._to_mono(str(final_path))

        meta: Dict[str, Any] = {
            "voice": voice,
            "lyrics": lyrics,
            "prompt": req.prompt,
            "mixed_with": instrumental_uri,
            "pitch_semitones": pitch_shift,
            "sample_rate": sample_rate,
            "duration_s": req.duration_s,
        }
        return str(final_path), meta

    # ---- helpers -------------------------------------------------------

    def _render_tts(self, text: str, voice: str, out: Path, sample_rate: int) -> None:
        """Try a local TTS CLI (piper/coqui), fall back to espeak-ng."""
        piper = shutil.which("piper")
        if piper:
            model = os.environ.get("ORBIT_PIPER_VOICE", voice)
            subprocess.run(
                [piper, "--model", model, "--output_file", str(out)],
                input=text.encode(),
                capture_output=True,
                check=True,
            )
            return

        espeak = shutil.which("espeak-ng")
        if espeak:
            subprocess.run(
                [espeak, "-v", voice, "-s", "150", "-w", str(out), text],
                check=True,
            )
            return

        raise RuntimeError("no TTS engine found (piper or espeak-ng)")

    def _pitch_shift(self, path: Path, semitones: float, sample_rate: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return
        factor = 2.0 ** (semitones / 12.0)
        tmp = path.with_suffix(".shift.wav")
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(path),
                "-filter:a", f"asetrate={int(sample_rate * factor)},"
                             f"aresample={sample_rate},atempo={1.0 / factor:.4f}",
                str(tmp),
            ],
            capture_output=True,
            check=True,
        )
        tmp.replace(path)

    def _mix(self, instrumental: str, vocal: str, out: str) -> None:
        """Sidechain-free ducked mix: instrumental at -6 dB under vocal at 0 dB."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg required for mixing")
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", instrumental,
                "-i", vocal,
                "-filter_complex",
                "[0:a]volume=0.5[bed];"
                "[1:a]volume=1.0[vox];"
                "[bed][vox]amix=inputs=2:duration=longest:dropout_transition=0,"
                "alimiter=limit=0.95",
                "-ar", "44100",
                out,
            ],
            capture_output=True,
            check=True,
        )

    def _to_mono(self, path: str) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return
        tmp = path + ".mono.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", path, "-ac", "1", tmp],
            capture_output=True,
            check=True,
        )
        shutil.move(tmp, path)
