"""Optional Ollama-compatible vision adapter for uploaded images."""
from __future__ import annotations
import base64
import json
import os
import urllib.request

from .media import UPLOAD_DIR, IMAGE_EXTENSIONS

def describe_image(file_id: str, prompt: str = "Describe this image in detail.") -> str:
    if not isinstance(file_id, str) or not file_id.isalnum() or len(file_id) != 32:
        raise ValueError("Invalid uploaded image identifier.")
    matches = [p for p in UPLOAD_DIR.glob(file_id + ".*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not matches:
        raise FileNotFoundError("Uploaded image not found.")
    endpoint = os.environ.get("ORBIT_VISION_URL", "http://127.0.0.1:11434/api/generate").strip()
    model = os.environ.get("ORBIT_VISION_MODEL", "").strip()
    if not model:
        raise RuntimeError("No vision model is configured. Set ORBIT_VISION_MODEL and ORBIT_VISION_URL to an Ollama-compatible vision endpoint.")
    data = matches[0].read_bytes()
    payload = json.dumps({
        "model": model,
        "prompt": str(prompt or "Describe this image in detail.")[:4000],
        "images": [base64.b64encode(data).decode("ascii")],
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read(2_000_000).decode("utf-8"))
    answer = result.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Vision endpoint returned no description.")
    return answer.strip()
