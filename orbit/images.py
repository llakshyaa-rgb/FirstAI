"""Image-generation adapter for a user-configured ComfyUI-compatible endpoint.

FirstAI itself is text-only. This adapter delegates pixel generation to a
separately installed image model/workflow exposed by ComfyUI's HTTP API.
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.parse
import urllib.request

class ImageBackendError(RuntimeError):
    pass

class ComfyUIImageBackend:
    def __init__(self, base_url="http://127.0.0.1:8188", timeout=30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method or ("POST" if data is not None else "GET"),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ImageBackendError(f"Image backend request failed: {exc}") from exc
        return raw, content_type

    def health(self):
        """Return ComfyUI system information or raise ImageBackendError."""
        raw, _ = self._request("/system_stats")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageBackendError("Image backend health response was not valid JSON") from exc

    def submit(self, workflow, client_id=None):
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("A non-empty ComfyUI API workflow dictionary is required")
        payload = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        raw, _ = self._request("/prompt", payload)
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImageBackendError("Image backend returned an invalid submission response") from exc
        if not result.get("prompt_id"):
            raise ImageBackendError(f"Image backend did not return prompt_id: {result}")
        return result["prompt_id"]

    def wait(self, prompt_id, poll_interval=2.0, timeout=600, on_progress=None):
        if poll_interval <= 0 or timeout <= 0:
            raise ValueError("poll_interval and timeout must be positive")
        deadline = time.monotonic() + timeout
        quoted = urllib.parse.quote(str(prompt_id), safe="")
        while time.monotonic() < deadline:
            raw, _ = self._request("/history/" + quoted)
            history = json.loads(raw.decode("utf-8"))
            item = history.get(str(prompt_id))
            if on_progress:
                on_progress({"kind": "status", "message": "The image model is rendering the requested artwork."})
            if item and item.get("outputs"):
                images = []
                for node_id, output in item["outputs"].items():
                    for image in output.get("images", []):
                        images.append({"node_id": node_id, **image})
                if images:
                    return images
            time.sleep(poll_interval)
        raise TimeoutError(f"Image generation timed out for job {prompt_id}")

    def download(self, image_info):
        required = ("filename", "subfolder", "type")
        if any(key not in image_info for key in required):
            raise ValueError("Image info must include filename, subfolder, and type")
        query = urllib.parse.urlencode({key: image_info[key] for key in required})
        raw, content_type = self._request("/view?" + query, method="GET")
        return raw, content_type

    def generate(self, workflow, poll_interval=2.0, timeout=600, on_progress=None):
        prompt_id = self.submit(workflow)
        if on_progress:
            on_progress({"kind": "status", "message": "The image request was accepted by ComfyUI."})
        images = self.wait(prompt_id, poll_interval=poll_interval, timeout=timeout,
                           on_progress=on_progress)
        return [{"info": item, "content": self.download(item)[0]} for item in images]
