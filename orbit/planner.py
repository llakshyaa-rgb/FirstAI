"""Strict JSON tool-plan validation and execution.

Only registered, non-confirmation tools can be called by model-generated plans.
Writes, code execution and other consequential actions remain owner-authorized.
"""
from __future__ import annotations
import json

MAX_STEPS = 5
AUTO_TOOLS = {"calculate", "web_search", "list_files", "read_file", "search_files"}

def parse_tool_plan(raw: str, registry):
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"steps"}:
        return None
    steps = payload["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        return None
    available = {item["name"]: item for item in registry.describe()}
    checked = []
    for step in steps:
        if not isinstance(step, dict) or set(step) != {"tool", "arguments"}:
            return None
        name, args = step["tool"], step["arguments"]
        if name not in AUTO_TOOLS or name not in available or available[name]["requires_confirmation"]:
            return None
        if not isinstance(args, dict) or len(args) > 8:
            return None
        checked.append((name, args))
    return checked

def execute_tool_plan(raw: str, registry, progress=None):
    steps = parse_tool_plan(raw, registry)
    if not steps:
        return None
    progress = progress or (lambda message: None)
    results = []
    for index, (name, arguments) in enumerate(steps, 1):
        progress(f"Executing planned step {index} of {len(steps)}: {name}.")
        try:
            result = registry.call(name, arguments)
            results.append({"step": index, "tool": name, "ok": True, "result": result})
        except Exception as exc:
            results.append({"step": index, "tool": name, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"})
            progress(f"Step {index} failed. Continuing with the remaining safe plan steps.")
    return results
