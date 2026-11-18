"""NSFW / safety scorer. Keyword-first, JSON-configurable, model-pluggable later."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "image_nsfw": {"threshold": 0.60, "keywords": [], "explicit_markers": []},
    "video_nsfw": {"threshold": 0.55, "keywords": [], "explicit_markers": []},
    "music_nsfw": {"threshold": 0.50, "keywords": [], "explicit_markers": []},
}


class SafetyScorer:
    """Score a prompt before dispatch. Caller decides what to do with score."""

    def __init__(
        self,
        kind: str,
        checkpoint: Optional[str] = None,
        model_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.kind = kind
        self.cfg = self._load(checkpoint or f"{kind}_nsfw.json")
        self._model_fn = model_fn

    def _load(self, filename: str) -> Dict[str, Any]:
        path = CHECKPOINT_DIR / filename
        try:
            with path.open() as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        cfg = dict(_DEFAULTS.get(self.kind, {"threshold": 0.5, "keywords": [], "explicit_markers": []}))
        cfg.update(data or {})
        return cfg

    def score_prompt(self, prompt: str) -> Tuple[float, List[str]]:
        """Return (score in [0,1], matched terms). Heuristic + optional model."""
        matched: List[str] = []

        for kw in self.cfg.get("keywords", []):
            if re.search(rf"\b{re.escape(kw)}\b", prompt, re.IGNORECASE):
                matched.append(kw)

        heuristic = min(1.0, len(matched) * 0.25)

        for marker in self.cfg.get("explicit_markers", []):
            if marker.lower() in prompt.lower():
                heuristic = min(1.0, heuristic + 0.3)
                matched.append(marker)

        score = heuristic
        if self._model_fn is not None:
            try:
                score = max(score, float(self._model_fn(prompt)))
            except Exception:  # noqa: BLE001
                pass  # fall back to heuristic silently

        return min(1.0, score), matched

    def is_safe(self, prompt: str) -> bool:
        score, _ = self.score_prompt(prompt)
        return score < float(self.cfg.get("threshold", 0.5))

    def check(self, prompt: str) -> Dict[str, Any]:
        """Convenience dict for logging / returning to callers."""
        score, matched = self.score_prompt(prompt)
        threshold = float(self.cfg.get("threshold", 0.5))
        return {
            "kind": self.kind,
            "score": score,
            "threshold": threshold,
            "safe": score < threshold,
            "matched": matched,
        }
