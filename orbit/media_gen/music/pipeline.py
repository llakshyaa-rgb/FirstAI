"""End-to-end music pipeline: safety → optional instrumental → optional vocal → mix."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..base import MediaKind, MediaRequest, MediaResult
from ..safety import SafetyScorer
from .base import MusicBackend, MusicResult


class MusicPipeline:
    def __init__(self, router=None) -> None:
        from ..router import route_request  # lazy to avoid cycle
        self._route = route_request
        self._safety = SafetyScorer("music_nsfw")
        self._router = router

    def run(
        self,
        prompt: str,
        duration_s: float = 30.0,
        seed: Optional[int] = None,
        instrumental_backend: str = "musicgen",
        vocal: Optional[Dict[str, Any]] = None,
    ) -> MusicResult:
        req = MediaRequest(
            kind=MediaKind.MUSIC,
            prompt=prompt,
            duration_s=duration_s,
            seed=seed,
        )

        safety = self._safety.check(prompt)
        if not safety["safe"]:
            return MusicResult(
                ok=False,
                request_id=req.request_id,
                kind=MediaKind.MUSIC,
                backend="pipeline",
                error=f"nsfw_score={safety['score']:.2f} matched={safety['matched']}",
            )

        instr = self._route(req, instrumental_backend)
        if not instr.ok:
            return MusicResult(
                ok=False,
                request_id=req.request_id,
                kind=MediaKind.MUSIC,
                backend="pipeline",
                error=f"instrumental failed: {instr.error}",
            )

        if not vocal:
            return instr

        vocal_req = MediaRequest(
            kind=MediaKind.MUSIC,
            prompt=prompt,
            duration_s=duration_s,
            seed=seed,
            extras={
                "instrumental_uri": instr.uri,
                "voice": vocal.get("voice", "default"),
                "lyrics": vocal.get("lyrics"),
                "pitch_semitones": vocal.get("pitch_semitones", 0.0),
            },
        )
        mixed = self._route(vocal_req, "vocal")
        if not mixed.ok:
            return instr  # fall back to instrumental-only
        return mixed
