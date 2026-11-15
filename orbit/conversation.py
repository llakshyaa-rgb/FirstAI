"""Small deterministic conversation layer for basic social exchanges.

This is deliberately narrow: it handles greetings and simple conversational
turns reliably, and returns None for substantive questions so FirstGPT can
answer them instead of pretending a template is general intelligence.
"""
from __future__ import annotations
import re


_GREETING = re.compile(
    r"^(?:hi|hey|hello|yo|yoo+|sup|wassup|what'?s up|good morning|good afternoon|good evening)"
    r"(?:\s+(?:orbit|there|dude|bro))?[!.?,\s]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(?:thanks?|thank you|thx|ty)(?:\s+.*)?[!.\s]*$", re.I)
_BYE = re.compile(r"^(?:bye|goodbye|see ya|see you|later|gn|good night)[!.\s]*$", re.I)
_HOW_ARE_YOU = re.compile(r"^(?:how are you|how're you|how are things|you good|how you doing)[?!\s.]*$", re.I)
_IDENTITY = re.compile(r"^(?:who are you|what are you|what is your name|what'?s your name)[?!\s.]*$", re.I)
_CAPABILITIES = re.compile(r"^(?:what can you do|what are your capabilities|how can you help me)[?!\s.]*$", re.I)


def quick_reply(message: str, history=None):
    """Return a concise grounded reply for basic social turns, else None."""
    text = re.sub(r"\s+", " ", str(message or "")).strip()
    if not text:
        return None
    recent = [str(item.get("content", "")) for item in (history or [])[-4:] if isinstance(item, dict)]
    if _GREETING.fullmatch(text):
        if any(_GREETING.fullmatch(item.strip()) for item in recent[-2:]):
            return "Hey again! What are we working on?"
        return "Yo! I'm here. What's up?"
    if _HOW_ARE_YOU.fullmatch(text):
        return "I'm here and ready to help. How are you doing?"
    if _THANKS.fullmatch(text):
        return "You're welcome! What else can I help with?"
    if _BYE.fullmatch(text):
        return "See you later! I'll be here when you want to pick this up."
    if _IDENTITY.fullmatch(text):
        return "I'm Orbit, your local assistant. I can chat, work with your files and workspace, and use the tools you've enabled."
    if _CAPABILITIES.fullmatch(text):
        return "I can help with questions, research, calculations, files, coding and project work. My local language model is still experimental, so I may need to be upfront when it can't answer reliably."
    return None
