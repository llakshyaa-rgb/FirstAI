"""Configurable prompt safety scoring with validated inputs."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "image_nsfw": {"threshold": 0.60, "keywords": ["explicit nudity", "pornographic", "sexualized"], "explicit_markers": []},
    "video_nsfw": {"threshold": 0.55, "keywords": ["explicit nudity", "pornographic", "sexualized"], "explicit_markers": []},
    "music_nsfw": {"threshold": 0.50, "keywords": ["explicit sexual content", "pornographic"], "explicit_markers": []},
}

class SafetyScorer:
    def __init__(self, kind: str, checkpoint: Optional[str] = None,
                 model_fn: Optional[Callable[[str], float]] = None) -> None:
        self.kind = kind
        self.cfg = self._load(checkpoint or f"{kind}_nsfw.json")
        self._model_fn = model_fn

    def _load(self, filename: str) -> Dict[str, Any]:
        path = CHECKPOINT_DIR / Path(filename).name
        try:
            with path.open(encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        cfg = dict(_DEFAULTS.get(self.kind, {"threshold": 0.5, "keywords": [], "explicit_markers": []}))
        cfg.update(data)
        try:
            threshold = float(cfg.get("threshold", 0.5))
        except (TypeError, ValueError):
            threshold = 0.5
        cfg["threshold"] = min(1.0, max(0.0, threshold)) if math.isfinite(threshold) else 0.5
        for key in ("keywords", "explicit_markers"):
            value = cfg.get(key, [])
            cfg[key] = [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []
        return cfg

    def score_prompt(self, prompt: str) -> Tuple[float, List[str]]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        matched: List[str] = []
        for keyword in self.cfg["keywords"]:
            if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", prompt, re.IGNORECASE):
                matched.append(keyword)
        score = min(1.0, len(matched) * 0.25)
        lowered = prompt.casefold()
        for marker in self.cfg["explicit_markers"]:
            if marker.casefold() in lowered:
                score = min(1.0, score + 0.3)
                matched.append(marker)
        if self._model_fn is not None:
            try:
                model_score = float(self._model_fn(prompt))
                if math.isfinite(model_score):
                    score = max(score, min(1.0, max(0.0, model_score)))
            except Exception:
                pass
        return score, matched

    def is_safe(self, prompt: str) -> bool:
        score, _ = self.score_prompt(prompt)
        return score < self.cfg["threshold"]

    def check(self, prompt: str) -> Dict[str, Any]:
        score, matched = self.score_prompt(prompt)
        threshold = self.cfg["threshold"]
        return {"kind": self.kind, "score": score, "threshold": threshold,
                "safe": score < threshold, "matched": matched}
