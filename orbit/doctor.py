"""Orbit environment diagnostics. Reports configuration without exposing secrets."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import urllib.request


def diagnose(checkpoint=None):
    checks = []
    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        add("PyTorch", True, torch.__version__)
        if cuda:
            props = torch.cuda.get_device_properties(0)
            add("CUDA GPU", True, f"{torch.cuda.get_device_name(0)} · {props.total_memory / 1024**3:.1f} GiB")
        else:
            add("CUDA GPU", False, "Not available; Orbit inference will use CPU.")
    except Exception as exc:
        add("PyTorch", False, f"{type(exc).__name__}: {exc}")

    target = Path(checkpoint or os.environ.get("ORBIT_CHECKPOINT", "checkpoints/firstgpt.best.pt"))
    add("Language checkpoint", target.is_file(), str(target))
    add("Brave image/web search", bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
        "API key configured" if os.environ.get("BRAVE_SEARCH_API_KEY", "").strip() else "BRAVE_SEARCH_API_KEY is not set.")
    workflow = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
    add("Image generation workflow", bool(workflow and Path(workflow).is_file()),
        workflow or "ORBIT_IMAGE_WORKFLOW is not set.")
    vision_model = os.environ.get("ORBIT_VISION_MODEL", "").strip()
    add("Vision model", bool(vision_model), vision_model or "ORBIT_VISION_MODEL is not set.")
    for package, label in (("playwright", "Playwright browser"), ("pypdfium2", "Scanned PDF renderer"),
                           ("faster_whisper", "Audio transcription"), ("pypdf", "PDF text extraction")):
        add(label, importlib.util.find_spec(package) is not None,
            "Installed" if importlib.util.find_spec(package) is not None else "Optional package not installed.")

    comfy_url = os.environ.get("ORBIT_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/") + "/system_stats"
    vision_endpoint = os.environ.get("ORBIT_VISION_URL", "http://127.0.0.1:11434/api/generate").rstrip("/")
    if os.environ.get("ORBIT_VISION_PROVIDER", "ollama").strip().lower() == "ollama":
        vision_url = vision_endpoint if vision_endpoint.endswith("/api/tags") else vision_endpoint.removesuffix("/api/generate").rstrip("/") + "/api/tags"
    else:
        vision_url = vision_endpoint if vision_endpoint.endswith("/v1/models") else vision_endpoint.removesuffix("/v1/chat/completions").rstrip("/") + "/v1/models"
    for name, url in (
        ("ComfyUI service", comfy_url),
        ("Vision service", vision_url),
    ):
        configured = bool(workflow) if name == "ComfyUI service" else bool(vision_model)
        if not configured:
            add(name, False, "Not configured; connection check skipped.")
            continue
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                add(name, response.status < 500, f"HTTP {response.status}")
        except Exception as exc:
            add(name, False, f"{type(exc).__name__}: {exc}")

    return checks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check Orbit's local dependencies and service configuration.")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    results = diagnose(args.checkpoint)
    for item in results:
        print(f"[{'OK' if item['ok'] else 'MISSING/UNAVAILABLE'}] {item['name']}: {item['detail']}")
    passed = sum(item["ok"] for item in results)
    print(f"\n{passed}/{len(results)} checks available. Missing optional services do not prevent text-only Orbit from starting.")


if __name__ == "__main__":
    main()
