"""Automatic intent routing for Orbit's single conversational interface.

This is deliberately a lightweight router: it chooses capabilities, not a
user-facing mode. Multiple capabilities may be selected for one request.
"""
from __future__ import annotations
import re


def _has(text: str, phrases) -> bool:
    return any(phrase in text for phrase in phrases)


def classify_request(message: str) -> dict:
    text = str(message or "").strip()
    lower = text.casefold()
    route = {"kind": "chat", "format": None, "prompt": text, "capabilities": []}

    # Explicit image generation is distinct from searching for existing images.
    image_verbs = ("generate", "draw", "create", "make", "design")
    image_nouns = ("image", "picture", "illustration", "wallpaper", "artwork")
    wants_image = any(re.search(r"\b" + re.escape(v) + r"\b", lower) for v in image_verbs) and any(
        re.search(r"\b" + re.escape(n) + r"\b", lower) for n in image_nouns
    )
    search_image = _has(lower, ("find images", "find image", "find pictures", "find picture",
                                "search images", "search pictures", "show me images", "show images",
                                "show pictures", "look for images", "look up images",
                                "image references", "photo references"))
    if wants_image and not search_image:
        route.update(kind="image_generation", prompt=text)
        route["capabilities"].append("image_generation")
        return route

    # Document/export requests go through the existing design engine.
    artifact_terms = (
        "pdf", "word document", "docx", "powerpoint", "pptx", "presentation", "slide deck",
        "spreadsheet", "excel", "xlsx", "csv", "report", "resume", "cv", "invoice",
        "ebook", "course document", "poster", "infographic", "diagram", "svg",
    )
    action_terms = ("make", "create", "build", "prepare", "write", "design", "generate", "export", "turn this into")
    has_artifact = any(term in lower for term in artifact_terms)
    has_action = any(term in lower for term in action_terms)
    if has_artifact and has_action and not _has(lower, ("website", "web app", "web application")):
        fmt = "pdf"
        for term, value in (("pptx", "pptx"), ("powerpoint", "pptx"), ("presentation", "pptx"),
                            ("slide deck", "pptx"), ("xlsx", "xlsx"), ("excel", "xlsx"),
                            ("spreadsheet", "xlsx"), ("csv", "csv"), ("docx", "docx"),
                            ("word document", "docx"), ("html", "html"), ("svg", "svg")):
            if term in lower:
                fmt = value
                break
        title = " ".join(text.split()[:8]).strip(" ,.:;!?") or "Orbit deliverable"
        route.update(kind="artifact", format=fmt, title=title, prompt=text)
        route["capabilities"].append("creative_studio")
        if _has(lower, ("research", "latest", "current", "recent", "look up", "sources", "fact check")):
            route["capabilities"].append("research")
        if _has(lower, ("calculate", "compute", "estimate", "compare")):
            route["capabilities"].append("analysis")
        return route

    if search_image:
        route["capabilities"].append("image_search")
    if _has(lower, ("research", "latest", "current", "recent", "look up", "search the web",
                    "search web", "find sources", "compare options", "fact check")):
        route["capabilities"].append("research")
    code_action = _has(lower, ("implement", "code", "coding", "debug", "fix", "refactor",
                               "modify", "edit the repo", "change the project", "build the app",
                               "add a feature", "integrate", "review this code"))
    code_subject = _has(lower, ("project", "repo", "repository", "codebase", "code", "app",
                                "application", "software", "script", "function", "bug", "error",
                                "website", "web app", "api", "backend", "frontend", "templateforge"))
    if code_action and code_subject:
        route["capabilities"].append("coding")
    if _has(lower, ("calculate", "compute", "equation", "percentage", "average", "total")) and re.search(r"\d", lower):
        route["capabilities"].append("analysis")
    if _has(lower, ("run tests", "test the project", "verify project", "validate project", "compile project")):
        route["capabilities"].append("verification")
    return route
