"""Specialist agents coordinated around one shared Orbit runtime.

Specialists use bounded concurrency and the same registered tools/runtime.
They return evidence and structured findings; Orbit remains the sole voice in chat.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import re
from .media_processing import process_upload
from .media import get_upload, search_web_images


@dataclass(frozen=True)
class Specialist:
    name: str
    description: str
    keywords: tuple[str, ...]
    tool_name: str | None = None


SPECIALISTS = (
    Specialist("researcher", "Find current public information and source links",
               ("research", "search the web", "search web", "look up", "latest", "current", "sources", "fact check"), "web_search"),
    Specialist("image_researcher", "Find existing image references and return source links",
               ("image_search", "image", "picture", "photo"), "image_search"),
    Specialist("code_scout", "Map relevant source files, definitions, dependencies and call sites",
               ("project", "repository", "repo", "codebase", "workspace", "file", "code", "bug", "error",
                "build", "implement", "create", "add", "fix", "debug", "modify", "refactor", "integrate"), "search_files"),
    Specialist("analyst", "Handle calculations, data checks and explicit numerical reasoning",
               ("calculate", "compute", "equation", "percentage", "average", "total", "compare", "estimate"), "calculate"),
    Specialist("verifier", "Run compile and test checks only when explicitly requested",
               ("verify project", "validate project", "run project tests", "test the project"), "compile_project"),
)


class AgentTeam:
    """Capability router and bounded specialist executor."""
    def __init__(self, registry, progress=None, max_workers=None):
        self.registry = registry
        self.progress = progress or (lambda message, **details: None)
        requested = int(max_workers or os.environ.get("ORBIT_AGENT_WORKERS", "4"))
        self.max_workers = max(1, min(5, requested))

    @staticmethod
    def _query_for(name, message):
        text = str(message or "").strip()
        if name == "researcher":
            match = re.search(r"(?:research|search(?: the)? web for|search web for|look up|find sources for|fact check)s+(.+)", text, re.I)
            return (match.group(1) if match else text)[:400]
        if name == "image_researcher":
            match = re.search(r"(?:find images|find pictures|search images|search pictures|show me images|look for images)\\s+(.+)", text, re.I)
            return (match.group(1) if match else text)[:350]
        if name == "code_scout":
            topics = ("image", "vision", "agent", "gpu", "memory", "training", "video", "audio",
                      "pdf", "browser", "tool", "server", "model", "chat", "ui", "task", "file",
                      "auth", "api", "database", "frontend", "backend", "template")
            found = [word for word in topics if word in text.casefold()]
            if found:
                return " ".join(found[:4])
            return text[:200]
        if name == "analyst":
            match = re.search(r"(?:calculate|compute|what is|what's|compare|estimate)s+(.+)", text, re.I)
            return (match.group(1) if match else text).replace("^", "**")[:180]
        return text

    def _run_specialist(self, specialist, message, attachments=None):
        if specialist.name == "media":
            results = []
            for file_id in (attachments or [])[:5]:
                try:
                    item = process_upload(file_id, "User request: " + message +
                        "\nAnalyse this attachment and extract details relevant to the request.")
                    results.append({"name": item["name"], "kind": item["kind"],
                                    "ok": True, "content": item["content"][:30000]})
                except Exception as exc:
                    fallback = {}
                    try:
                        stored = get_upload(file_id)
                        if stored.get("text"):
                            fallback = {"fallback_text": stored["text"][:30000]}
                    except Exception:
                        pass
                    results.append({"file_id": file_id, "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}", **fallback})
            return results

        query = self._query_for(specialist.name, message)
        if specialist.tool_name == "web_search":
            return self.registry.call("web_search", {"query": query, "count": 8})
        if specialist.tool_name == "image_search":
            explicit_mature = any(term in message.casefold() for term in ("mature", "adult", "nsfw", "18+"))
            return search_web_images(query, count=8, safesearch="off" if explicit_mature else "strict")
        if specialist.tool_name == "search_files":
            files = self.registry.call("list_files", {"relative": "."})
            matches = self.registry.call("search_files", {"query": query})
            return {"project_files": files[:120], "matching_lines": matches[:80],
                    "search_term": query,
                    "guidance": "Map call sites and dependencies before edits; prefer minimal targeted patches."}
        if specialist.tool_name == "compile_project":
            compiled = self.registry.call("compile_project", confirmed=True)
            if not compiled.get("passed"):
                return {"compile": compiled, "tests": "not run because compilation failed"}
            tests = self.registry.call("run_tests", confirmed=True)
            return {"compile": compiled, "tests": tests}
        if specialist.tool_name == "calculate":
            return self.registry.call("calculate", {"expression": query})
        raise ValueError("Specialist has no executable tool")

    def run(self, message, attachments=None, capabilities=None):
        text = str(message or "")
        lower = text.casefold()
        requested = set(capabilities or ())
        selected = []
        for specialist in SPECIALISTS:
            choose = (
                (specialist.name == "researcher" and ("research" in requested or any(x in lower for x in ("research", "search the web", "search web", "look up", "latest", "current", "fact check"))))
                or (specialist.name == "image_researcher" and "image_search" in requested)
                or (specialist.name == "code_scout" and ("coding" in requested or "code_scout" in requested))
                or (specialist.name == "analyst" and ("analysis" in requested or any(x in lower for x in ("calculate", "compute", "equation", "percentage", "average", "total")) and any(ch.isdigit() for ch in text)))
                or (specialist.name == "verifier" and ("verification" in requested or any(x in lower for x in ("verify project", "validate project", "run project tests", "test the project"))))
            )
            if choose:
                selected.append(specialist)
        if attachments:
            selected.append(Specialist("media", "Read and interpret owner-supplied attachments", (), None))
        if not selected:
            return []
        self.progress("Orbit's coordinator assigned " + ", ".join(item.name for item in selected) +
                      ". Independent specialists are working in parallel; Orbit will combine their findings.")
        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected)),
                                thread_name_prefix="orbit-agent") as pool:
            futures = {pool.submit(self._run_specialist, item, text, attachments): item for item in selected}
            for future in as_completed(futures):
                specialist = futures[future]
                try:
                    results.append({"agent": specialist.name, "ok": True, "result": future.result()})
                except Exception as exc:
                    self.progress(f"{specialist.name} agent failed ({type(exc).__name__}: {exc}); continuing with other specialists.",
                                  kind="method_failed", agent=specialist.name)
                    results.append({"agent": specialist.name, "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
        return sorted(results, key=lambda item: next(
            (index for index, specialist in enumerate(selected) if specialist.name == item["agent"]), 99))
