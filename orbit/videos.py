"""ComfyUI API adapter for text-to-video workflows and video output nodes."""
from __future__ import annotations
import copy
import json
import math
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class VideoBackendError(RuntimeError):
    pass


def load_video_workflow_template(path, prompt, negative_prompt="", seed=None,
                                 width=576, height=576, frames=49, fps=12,
                                 steps=20, cfg=6.0):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Video prompt must be non-empty text")
    if not isinstance(negative_prompt, str):
        raise ValueError("Negative prompt must be text")
    if type(seed) is not int and seed is not None:
        raise ValueError("Seed must be an integer")
    if seed is not None and not 0 <= seed <= 2**64 - 1:
        raise ValueError("Seed must be between 0 and 2^64-1")
    if any(type(v) is not int for v in (width, height, frames, fps, steps)):
        raise ValueError("Video dimensions, frames, fps and steps must be integers")
    if not (64 <= width <= 2048 and 64 <= height <= 2048 and width % 8 == 0 and height % 8 == 0):
        raise ValueError("Video dimensions must be multiples of 8 between 64 and 2048")
    if not 2 <= frames <= 500 or not 1 <= fps <= 60 or not 1 <= steps <= 100:
        raise ValueError("Video frames, fps or steps are outside supported limits")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(float(cfg)) or not 0 <= cfg <= 30:
        raise ValueError("CFG must be a finite number between 0 and 30")
    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Video workflow must be a non-empty ComfyUI API workflow object")
    values = {
        "{{PROMPT}}": prompt, "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{SEED}}": seed if seed is not None else secrets.randbits(64),
        "{{WIDTH}}": width, "{{HEIGHT}}": height, "{{FRAMES}}": frames,
        "{{FPS}}": fps, "{{STEPS}}": steps, "{{CFG}}": float(cfg),
    }
    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            if value in values:
                return values[value]
            for marker, replacement in values.items():
                if isinstance(replacement, str):
                    value = value.replace(marker, replacement)
        return value
    return visit(copy.deepcopy(workflow))


class ComfyUIVideoBackend:
    def __init__(self, base_url="http://127.0.0.1:8188", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method or ("POST" if data is not None else "GET"),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise VideoBackendError(f"Video backend request failed: {exc}") from exc

    def submit(self, workflow):
        raw = self._request("/prompt", {"prompt": workflow})
        result = json.loads(raw.decode("utf-8"))
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise VideoBackendError(f"ComfyUI did not return prompt_id: {result}")
        return prompt_id

    def wait(self, prompt_id, timeout=1800, poll_interval=2, on_progress=None):
        deadline = time.monotonic() + timeout
        quoted = urllib.parse.quote(str(prompt_id), safe="")
        ticks = 0
        while time.monotonic() < deadline:
            history = json.loads(self._request("/history/" + quoted).decode("utf-8"))
            item = history.get(str(prompt_id))
            ticks += 1
            if on_progress and ticks in (1, 6, 16, 36, 90, 180):
                on_progress({"kind": "status", "message": "Video render is running on the connected GPU. Longer clips can take several minutes."})
            if item:
                status = item.get("status", {})
                messages = status.get("messages", [])
                failures = [entry for entry in messages if isinstance(entry, list) and len(entry) > 1 and entry[1] == "execution_error"]
                if status.get("status_str") == "error" or failures:
                    raise VideoBackendError(f"ComfyUI video workflow failed: {failures or status}")
                outputs = []
                for node_id, output in item.get("outputs", {}).items():
                    for key in ("videos", "gifs"):
                        for info in output.get(key, []):
                            outputs.append({"node_id": node_id, **info})
                if outputs:
                    return outputs
                if status.get("completed"):
                    raise VideoBackendError("Workflow finished but returned no video/gif outputs. Add a video output node and export an API workflow.")
            time.sleep(poll_interval)
        raise TimeoutError(f"Video generation timed out for job {prompt_id}")

    def download(self, info):
        if any(key not in info for key in ("filename", "subfolder", "type")):
            raise ValueError("Video output metadata is incomplete")
        query = urllib.parse.urlencode({key: info[key] for key in ("filename", "subfolder", "type")})
        return self._request("/view?" + query, method="GET")

    def generate(self, workflow, timeout=1800, on_progress=None):
        prompt_id = self.submit(workflow)
        if on_progress:
            on_progress({"kind": "status", "message": "Video workflow accepted. Waiting for rendered frames and encoding."})
        return [{**info, "content": self.download(info)}
                for info in self.wait(prompt_id, timeout=timeout, on_progress=on_progress)]
