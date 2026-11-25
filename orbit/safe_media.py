"""Conservative prompt guard for Orbit's youth-accessible creative media UI."""
from __future__ import annotations
import re

_BLOCKED = (
    r"\bnude\b", r"\bnudity\b", r"\bnaked\b", r"\btopless\b",
    r"\bgenitals?\b", r"\bpenis\b", r"\bvagina\b", r"\bporn(?:ography)?\b",
    r"\berotic\b", r"\bsexually explicit\b", r"\bsex act\b", r"\bnsfw\b",
)

def validate_media_prompt(prompt: str) -> None:
    """Reject explicit sexual/nudity requests; permit non-explicit fashion/art."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty creative prompt is required.")
    lowered = prompt.casefold()
    if any(re.search(pattern, lowered) for pattern in _BLOCKED):
        raise ValueError(
            "Orbit's image and video tools support non-explicit creative work only. "
            "Try editorial fashion, swimwear or lingerie design, or a non-explicit classical figure study."
        )
