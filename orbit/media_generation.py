"""Generic ComfyUI media generation for video and audio workflows.

Orbit owns the chat/task flow; ComfyUI owns the selected specialist model graph.
Workflow JSON files are configured separately and use {{PROMPT}}, {{NEGATIVE_PROMPT}},
{{SEED}}, {{WIDTH}}, {{HEIGHT}}, {{FPS}}, {{FRAMES}}, and {{DURATION}} placeholders.
"""
from __future__ import annotations
import copy
import json
import secrets
import time
import urllib.parse
import urllib.request
from pathlib import Path


class MediaGenerationError(RuntimeError):
    pass


def load_media_workflow(path, prompt, kind, negative_prompt="", seed=None):
    if kind not in {"video", "audio"}:
        raise ValueError("Media kind must be video or audio.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty generation prompt is required.")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**64 - 1):
        raise ValueError("Seed must be an integer between 0 and 2^64-1.")
    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("ComfyUI workflow must be a non-empty API-format JSON object.")
    values = {
        "{{PROMPT}}": prompt,
        "{{NEGATIVE_PROMPT}}": str(negative_prompt or ""),
        "{{SEED}}": seed if seed is not None else secrets.randbits(64),
        "{{WIDTH}}": int(__import__("os").environ.get("ORBIT_VIDEO_WIDTH", "832")),
        "{{HEIGHT}}": int(__import__("os").environ.get("ORBIT_VIDEO_HEIGHT", "480")),
        "{{FPS}}": int(__import__("os").environ.get("ORBIT_VIDEO_FPS", "16")),
        "{{FRAMES}}": int(__import__("os").environ.get("ORBIT_VIDEO_FRAMES", "49")),
        "{{DURATION}}": float(__import__("os").environ.get("ORBIT_AUDIO_DURATION", "15")),
    }
    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            if value in values:
                return values[value]
            for marker in ("{{PROMPT}}", "{{NEGATIVE_PROMPT}}"):
                value = value.replace(marker, values[marker])
        return value
    return replace(copy.deepcopy(workflow))


class ComfyUIMediaBackend:
    def __init__(self, base_url="http://127.0.0.1:8188", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except Exception as exc:
            raise MediaGenerationError(f"ComfyUI request failed: {type(exc).__name__}: {exc}") from exc

    def generate(self, workflow, kind, timeout=1800, on_progress=None):
        prompt_result = json.loads(self._request("/prompt", {"prompt": workflow}).decode("utf-8"))
        prompt_id = prompt_result.get("prompt_id")
        if not prompt_id:
            raise MediaGenerationError(f"ComfyUI did not accept the workflow: {prompt_result}")
        if on_progress:
            on_progress({"kind": "status", "message": f"The {kind} specialist accepted the request."})
        keys = ("videos", "gifs") if kind == "video" else ("audio",)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = json.loads(self._request("/history/" + urllib.parse.quote(prompt_id, safe="")).decode("utf-8"))
            item = history.get(prompt_id)
            if item:
                status = item.get("status", {})
                failures = [entry for entry in status.get("messages", [])
                            if isinstance(entry, list) and len(entry) > 1 and entry[1] == "execution_error"]
                if status.get("status_str") == "error" or failures:
                    raise MediaGenerationError(f"ComfyUI {kind} workflow failed: {failures or status}")
                outputs = []
                for node_id, node_output in (item.get("outputs") or {}).items():
                    for key in keys:
                        for media in node_output.get(key, []) or []:
                            outputs.append({"node_id": node_id, "type": key, **media})
                if outputs:
                    results = []
                    for info in outputs:
                        required = ("filename", "subfolder", "type")
                        if any(field not in info for field in required):
                            continue
                        query = urllib.parse.urlencode({field: info[field] for field in required})
                        raw = self._request("/view?" + query)
                        results.append({"info": info, "content": raw})
                    if results:
                        return results
                if status.get("completed"):
                    raise MediaGenerationError(f"ComfyUI completed the {kind} workflow but returned no {kind} outputs.")
            time.sleep(2)
        raise TimeoutError(f"ComfyUI {kind} generation timed out after {timeout} seconds.")
