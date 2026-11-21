"""End-to-end music generation: instrumental track plus optional vocal mix."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..base import MediaKind, MediaRequest
from .base import MusicResult


class MusicPipeline:
    def __init__(self, router=None) -> None:
        """Accept a Router instance; otherwise use Orbit's default router."""
        if router is None:
            from ..router import route_request

            self._route = route_request
        elif callable(getattr(router, "route", None)):
            self._route = router.route
        elif callable(router):
            self._route = router
        else:
            raise TypeError("router must be callable or expose route()")

    @staticmethod
    def _failure(request: MediaRequest, error: str) -> MusicResult:
        return MusicResult(
            ok=False,
            request_id=request.request_id,
            kind=MediaKind.MUSIC,
            backend="pipeline",
            error=error,
        )

    def run(
        self,
        prompt: str,
        duration_s: float = 30.0,
        seed: Optional[int] = None,
        instrumental_backend: str = "musicgen",
        vocal: Optional[Dict[str, Any]] = None,
    ) -> MusicResult:
        if not isinstance(prompt, str) or not prompt.strip():
            request = MediaRequest(
                kind=MediaKind.MUSIC,
                prompt=str(prompt or ""),
                duration_s=duration_s,
                seed=seed,
            )
            return self._failure(request, "music prompt cannot be empty")

        if duration_s <= 0:
            request = MediaRequest(
                kind=MediaKind.MUSIC,
                prompt=prompt,
                duration_s=duration_s,
                seed=seed,
            )
            return self._failure(request, "duration_s must be greater than zero")

        request = MediaRequest(
            kind=MediaKind.MUSIC,
            prompt=prompt,
            duration_s=duration_s,
            seed=seed,
        )

        lyrics = None
        if vocal is not None:
            lyrics = vocal.get("lyrics")
            if lyrics is not None and not isinstance(lyrics, str):
                return self._failure(request, "vocal lyrics must be a string")

        instrumental = self._route(request, instrumental_backend)
        if not instrumental.ok:
            return self._failure(
                request,
                f"instrumental generation failed: {instrumental.error}",
            )

        if not vocal:
            return instrumental

        vocal_request = MediaRequest(
            kind=MediaKind.MUSIC,
            prompt=prompt,
            duration_s=duration_s,
            seed=seed,
            extras={
                "instrumental_uri": instrumental.uri,
                "voice": vocal.get("voice", "default"),
                "lyrics": lyrics,
                "pitch_semitones": vocal.get("pitch_semitones", 0.0),
                "sample_rate": vocal.get("sample_rate", 44100),
                "channels": vocal.get("channels", 1),
            },
        )

        mixed = self._route(vocal_request, "vocal")
        if not mixed.ok:
            return self._failure(
                request,
                f"vocal generation or mixing failed: {mixed.error}",
            )

        mixed.meta = {
            **(instrumental.meta or {}),
            **(mixed.meta or {}),
            "instrumental_uri": instrumental.uri,
            "vocal_mix": True,
        }
        return mixed
