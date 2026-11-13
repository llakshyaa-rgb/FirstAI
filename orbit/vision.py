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

def _vision_image_bytes(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg"
    try:
        if suffix in {".heic", ".heif"}:
            import pillow_heif
            pillow_heif.register_heif_opener()
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("This image format needs Pillow conversion support. Install requirements-media.txt for HEIC/HEIF/AVIF and other image conversion.") from exc
    try:
        with Image.open(path) as image:
            image.seek(0)
            converted = image.convert("RGB")
            import io
            output = io.BytesIO()
            converted.save(output, format="PNG")
            return output.getvalue(), "image/png"
    except Exception as exc:
        raise RuntimeError("Could not convert " + (suffix or "image") + " for the configured vision model: " + str(exc)) from exc


def describe_image(file_id: str, prompt: str = "Describe this image in detail.") -> str:
    path = _image_path(file_id)
    provider = os.environ.get("ORBIT_VISION_PROVIDER", "ollama").strip().lower()
    endpoint = os.environ.get("ORBIT_VISION_URL", "http://127.0.0.1:11434/api/generate").strip()
    model = os.environ.get("ORBIT_VISION_MODEL", "").strip()
    if not model:
        raise RuntimeError("No vision model is configured. Set ORBIT_VISION_MODEL and ORBIT_VISION_URL.")
    image_bytes, mime = _vision_image_bytes(path)
    encoded = base64.b64encode(image_bytes).decode("ascii")
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
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.loads(response.read(4_000_000).decode("utf-8"))
    except Exception as primary_error:
        fallback_endpoint = os.environ.get("ORBIT_VISION_FALLBACK_URL", "").strip()
        if not fallback_endpoint or fallback_endpoint == endpoint:
            raise
        fallback_model = os.environ.get("ORBIT_VISION_FALLBACK_MODEL", model).strip()
        fallback_payload = dict(payload)
        fallback_payload["model"] = fallback_model
        if provider == "ollama":
            fallback_url = fallback_endpoint if fallback_endpoint.endswith("/api/generate") else fallback_endpoint.rstrip("/") + "/api/generate"
        else:
            fallback_url = fallback_endpoint if fallback_endpoint.rstrip("/").endswith("/chat/completions") else fallback_endpoint.rstrip("/") + "/v1/chat/completions"
        fallback_request = urllib.request.Request(
            fallback_url, data=json.dumps(fallback_payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(fallback_request, timeout=240) as response:
                result = json.loads(response.read(4_000_000).decode("utf-8"))
        except Exception as fallback_error:
            raise RuntimeError(
                f"Primary vision backend failed: {primary_error}; fallback vision backend also failed: {fallback_error}"
            ) from fallback_error
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
