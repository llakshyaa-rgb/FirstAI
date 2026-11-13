"""Bounded specialist-agent team coordinated around one shared Orbit runtime.

The coordinator delegates independent read-only work concurrently. The media
specialist handles owner-supplied attachments without loading another LLM.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import re
from .media_processing import process_upload
from .media import get_upload


@dataclass(frozen=True)
class Specialist:
    name: str
    description: str
    keywords: tuple[str, ...]
    tool_name: str | None = None


SPECIALISTS = (
    Specialist("researcher", "Find current public information and source links",
               ("research", "search the web", "search web", "look up", "latest", "current", "sources", "website"), "web_search"),
    Specialist("workspace", "Scout existing project files and implementation patterns",
               ("project", "repository", "repo", "codebase", "workspace", "file", "code", "bug", "error",
                "build", "implement", "create", "add", "fix", "debug", "modify", "refactor", "integrate"), "search_files"),
    Specialist("analyst", "Handle explicit calculations and numerical checks",
               ("calculate", "compute", "equation", "percentage", "average", "total"), "calculate"),
    Specialist("verifier", "Run requested project checks after implementation",
               ("verify project", "validate project", "run project tests", "test the project"), "compile_project"),
)


class AgentTeam:
    """Coordinator assigns only the specialists relevant to the owner's request."""
    def __init__(self, registry, progress=None, max_workers=None):
        self.registry = registry
        self.progress = progress or (lambda message, **details: None)
        requested = int(max_workers or os.environ.get("ORBIT_AGENT_WORKERS", "3"))
        self.max_workers = max(1, min(4, requested))

    @staticmethod
    def _query_for(name, message):
        text = message.strip()
        if name == "researcher":
            match = re.search(r"(?:research|search(?: the)? web for|search web for|look up|find sources for)\s+(.+)", text, re.I)
            return (match.group(1) if match else text)[:300]
        if name == "workspace":
            match = re.search(r"(?:search (?:the )?(?:project|workspace|repo(?:sitory)?) for|find in (?:the )?(?:project|repo(?:sitory)?)|search files for)\s+(.+)", text, re.I)
            if match:
                return match.group(1)[:200]
            topics = ("image", "vision", "agent", "gpu", "memory", "training", "video", "audio",
                      "pdf", "browser", "tool", "server", "model", "chat", "ui", "task", "file")
            found = next((word for word in topics if word in text.casefold()), None)
            return found or "orbit"
        if name == "analyst":
            match = re.search(r"(?:calculate|compute|what is|what's)\s+(.+)", text, re.I)
            return (match.group(1) if match else text).replace("^", "**")[:160]
        return text

    def _run_specialist(self, specialist, message, attachments=None):
        if specialist.name == "media":
            results = []
            for file_id in (attachments or [])[:5]:
                try:
                    item = process_upload(
                        file_id,
                        "User request: " + message + "\nAnalyse this attachment and extract the details relevant to the request."
                    )
                    results.append({
                        "name": item["name"], "kind": item["kind"],
                        "ok": True, "content": item["content"][:30000],
                    })
                except Exception as exc:
                    fallback = {}
                    try:
                        stored = get_upload(file_id)
                        if stored.get("text"):
                            fallback = {"fallback_text": stored["text"][:30000]}
                    except Exception:
                        pass
                    results.append({
                        "file_id": file_id, "ok": False,
                        "error": f"{type(exc).__name__}: {exc}", **fallback,
                    })
            return results

        query = self._query_for(specialist.name, message)
        if specialist.tool_name == "web_search":
            return self.registry.call("web_search", {"query": query, "count": 6})
        if specialist.tool_name == "search_files":
            files = self.registry.call("list_files", {"relative": "."})
            matches = self.registry.call("search_files", {"query": query})
            return {"project_files": files[:100], "matching_lines": matches[:50], "search_term": query}
        if specialist.tool_name == "compile_project":
            compiled = self.registry.call("compile_project", confirmed=True)
            if not compiled.get("passed"):
                return {"compile": compiled, "tests": "not run because compilation failed"}
            tests = self.registry.call("run_tests", confirmed=True)
            return {"compile": compiled, "tests": tests}
        if specialist.tool_name == "calculate":
            return self.registry.call("calculate", {"expression": query})
        raise ValueError("Specialist has no executable tool")

    def run(self, message, attachments=None):
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
                    phrase in lower for phrase in ("search", "find", "inspect", "check", "bug", "error",
                                                   "build", "implement", "create", "add", "fix", "debug",
                                                   "modify", "refactor", "integrate")
                ):
                    continue
                if specialist.tool_name == "compile_project" and not any(
                    phrase in lower for phrase in ("verify project", "validate project", "run project tests", "test the project")
                ):
                    continue
                if specialist.tool_name == "calculate" and not any(ch.isdigit() for ch in text):
                    continue
                selected.append(specialist)

        if attachments:
            selected.append(Specialist("media", "Read and interpret owner-supplied files", (), None))
        if not selected:
            return []

        self.progress(
            "Coordinator assigned " + ", ".join(item.name for item in selected) +
            " specialist agent(s). Independent work runs concurrently where possible."
        )
        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected)),
                                thread_name_prefix="orbit-agent") as pool:
            futures = {pool.submit(self._run_specialist, item, text, attachments): item for item in selected}
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
