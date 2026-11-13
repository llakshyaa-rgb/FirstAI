"""Vision adapters for Orbit image understanding.

Supports Ollama's /api/generate image format and OpenAI-compatible
/chat/completions servers that expose a vision-capable model.
"""
from __future__ import annotations
import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from .media import UPLOAD_DIR, IMAGE_EXTENSIONS

def _image_path(file_id: str) -> Path:
    if not isinstance(file_id, str) or len(file_id) != 32 or not all(c in "0123456789abcdef" for c in file_id.lower()):
        raise ValueError("Invalid uploaded image identifier.")
    matches = [p for p in UPLOAD_DIR.glob(file_id + ".*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not matches:
        raise FileNotFoundError("Uploaded image not found.")
    path = matches[0].resolve()
    if UPLOAD_DIR.resolve() not in path.parents:
        raise ValueError("Image path is outside the upload library.")
    return path

def describe_image(file_id: str, prompt: str = "Describe this image in detail.") -> str:
    path = _image_path(file_id)
    provider = os.environ.get("ORBIT_VISION_PROVIDER", "ollama").strip().lower()
    endpoint = os.environ.get("ORBIT_VISION_URL", "http://127.0.0.1:11434/api/generate").strip()
    model = os.environ.get("ORBIT_VISION_MODEL", "").strip()
    if not model:
        raise RuntimeError("No vision model is configured. Set ORBIT_VISION_MODEL and ORBIT_VISION_URL.")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    question = str(prompt or "Describe this image in detail.")[:4000]
    if provider == "ollama":
        url = endpoint if endpoint.endswith("/api/generate") else endpoint.rstrip("/") + "/api/generate"
        payload = {"model": model, "prompt": question, "images": [encoded], "stream": False}
        headers = {"Content-Type": "application/json"}
    elif provider in {"openai", "openai-compatible", "llamacpp"}:
        url = endpoint if endpoint.rstrip("/").endswith("/chat/completions") else endpoint.rstrip("/") + "/v1/chat/completions"
        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "stream": False, "max_tokens": 1200}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("ORBIT_VISION_API_KEY", "").strip()
        if key:
            headers["Authorization"] = "Bearer " + key
    else:
        raise ValueError("ORBIT_VISION_PROVIDER must be ollama or openai-compatible.")
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read(4_000_000).decode("utf-8"))
    if provider == "ollama":
        answer = result.get("response")
    else:
        choices = result.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        answer = message.get("content")
        if isinstance(answer, list):
            answer = "\n".join(str(part.get("text", "")) for part in answer if isinstance(part, dict))
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Vision endpoint returned no description.")
    return answer.strip()
