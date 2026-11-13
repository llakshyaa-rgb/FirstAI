"""Lightweight specialist-agent team for Orbit.

Specialists share one model runtime; they do not load duplicate LLM weights.
Independent read-only tools run concurrently, then one synthesis call produces
the user-facing answer. This avoids extra model calls and GPU-memory duplication.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import re
import threading
import time


@dataclass(frozen=True)
class Specialist:
    name: str
    description: str
    keywords: tuple[str, ...]
    tool_name: str | None = None


SPECIALISTS = (
    Specialist("researcher", "Find current public information and source links",
               ("research", "search the web", "search web", "look up", "latest", "current", "sources", "website"), "web_search"),
    Specialist("workspace", "Inspect project files and local technical context",
               ("project", "repository", "repo", "codebase", "workspace", "file", "code", "bug", "error"), "search_files"),
    Specialist("analyst", "Handle explicit calculations and numerical checks",
               ("calculate", "compute", "equation", "percentage", "average", "total"), "calculate"),
)


class AgentTeam:
    """Routes relevant specialist tools concurrently, with bounded overhead."""
    def __init__(self, registry, progress=None, max_workers=None):
        self.registry = registry
        self.progress = progress or (lambda message, **details: None)
        requested = int(max_workers or os.environ.get("ORBIT_AGENT_WORKERS", "3"))
        self.max_workers = max(1, min(4, requested))
        self._lock = threading.Lock()

    @staticmethod
    def _query_for(name, message):
        text = message.strip()
        if name == "researcher":
            match = re.search(r"(?:research|search(?: the)? web for|search web for|look up|find sources for)\s+(.+)", text, re.I)
            return (match.group(1) if match else text)[:300]
        if name == "workspace":
            match = re.search(r"(?:search (?:the )?(?:project|workspace|repo(?:sitory)?) for|find in (?:the )?(?:project|repo(?:sitory)?)|search files for)\s+(.+)", text, re.I)
            return (match.group(1) if match else text)[:200]
        if name == "analyst":
            match = re.search(r"(?:calculate|compute|what is|what's)\s+(.+)", text, re.I)
            return (match.group(1) if match else text).replace("^", "**")[:160]
        return text

    def _run_specialist(self, specialist, message):
        query = self._query_for(specialist.name, message)
        if specialist.tool_name == "web_search":
            return self.registry.call("web_search", {"query": query, "count": 6})
        if specialist.tool_name == "search_files":
            return self.registry.call("search_files", {"query": query})
        if specialist.tool_name == "calculate":
            return self.registry.call("calculate", {"expression": query})
        raise ValueError("Specialist has no executable tool")

    def run(self, message):
        text = str(message or "")
        lower = text.casefold()
        selected = []
        for specialist in SPECIALISTS:
            if any(word in lower for word in specialist.keywords):
                if specialist.tool_name == "web_search" and not any(
                    phrase in lower for phrase in ("research", "search the web", "search web", "look up", "latest", "current", "sources")
                ):
                    continue
                if specialist.tool_name == "search_files" and not any(
                    phrase in lower for phrase in ("search", "find", "inspect", "check", "bug", "error")
                ):
                    continue
                if specialist.tool_name == "calculate" and not any(ch.isdigit() for ch in text):
                    continue
                selected.append(specialist)

        if not selected:
            return []

        self.progress(
            "Coordinator assigned " + ", ".join(item.name for item in selected) +
            " specialist agent(s). Independent checks are running in parallel."
        )
        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected)),
                                thread_name_prefix="orbit-agent") as pool:
            futures = {pool.submit(self._run_specialist, item, text): item for item in selected}
            for future in as_completed(futures):
                specialist = futures[future]
                try:
                    value = future.result()
                    results.append({"agent": specialist.name, "ok": True, "result": value})
                except Exception as exc:
                    self.progress(
                        f"{specialist.name.capitalize()} agent failed ({type(exc).__name__}: {exc}); "
                        "the coordinator will continue with the other results.",
                        kind="method_failed", agent=specialist.name,
                    )
                    results.append({"agent": specialist.name, "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
        return sorted(results, key=lambda item: next(
            (index for index, specialist in enumerate(selected) if specialist.name == item["agent"]), 99
        ))
