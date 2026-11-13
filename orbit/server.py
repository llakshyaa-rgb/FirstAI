"""Local Orbit chat server with SQLite history and live task events.

Run: python -m orbit.server --checkpoint checkpoints/firstgpt.best.pt
This uses only Python's standard library plus the project's PyTorch dependency.
"""
from __future__ import annotations
import argparse
import json
import mimetypes
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import threading
import uuid

from .runtime import OrbitRuntime, ModelNotReady
from .tasks import TaskStore
from .memory import MemoryStore
from .images import ComfyUIImageBackend
from .image_workflow import load_workflow_template
from .agent import LocalAgent

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
GENERATED = ROOT / "generated"

class OrbitApplication:
    def __init__(self, checkpoint, state_path):
        self.runtime = OrbitRuntime(checkpoint)
        self.store = TaskStore(state_path)
        self.memory = MemoryStore(state_path)
        self._threads = set()
        self._lock = threading.Lock()
        # Resume interrupted chat-generation tasks after the checkpoint is loaded.
        for task_id in self.store.recover_interrupted():
            task = self.store.get(task_id)
            if task and task["goal"].startswith("Generate image: "):
                self.store.update(task_id, "needs_input", error="Image generation was interrupted by a server restart. Resubmit the image request.")
                self.store.emit(task_id, "method_failed", "The image job was interrupted by a restart. Its prompt is preserved; please resubmit it.")
            elif task and self.runtime.ready:
                self.store.update(task_id, "queued")
                self._launch_chat(task_id, task["goal"])

    def start_chat(self, message):
        task_id = self.store.create(message)
        self._launch_chat(task_id, message)
        return task_id

    def start_image(self, prompt, negative_prompt="", seed=None):
        task_id = self.store.create("Generate image: " + prompt)
        thread = threading.Thread(target=self._image_worker, args=(task_id, prompt, negative_prompt, seed), daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()
        return task_id

    def _image_worker(self, task_id, prompt, negative_prompt, seed):
        try:
            workflow_path = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
            if not workflow_path or not Path(workflow_path).is_file():
                raise RuntimeError("Image workflow is not configured. Set ORBIT_IMAGE_WORKFLOW to a ComfyUI API workflow JSON file.")
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "The image backend is preparing the requested artwork.")
            workflow = load_workflow_template(
                workflow_path, prompt, negative_prompt=negative_prompt, seed=seed,
                checkpoint=os.environ.get("ORBIT_IMAGE_CHECKPOINT", "model.safetensors"),
                width=int(os.environ.get("ORBIT_IMAGE_WIDTH", "768")),
                height=int(os.environ.get("ORBIT_IMAGE_HEIGHT", "768")),
                steps=int(os.environ.get("ORBIT_IMAGE_STEPS", "24")),
                cfg=float(os.environ.get("ORBIT_IMAGE_CFG", "7.0")),
            )
            backend = ComfyUIImageBackend(base_url=os.environ.get("ORBIT_COMFYUI_URL", "http://127.0.0.1:8188"))
            progress = lambda event: self.store.emit(
                task_id, event["kind"], event["message"],
                **{key: value for key, value in event.items() if key not in ("kind", "message")}
            )
            try:
                outputs = backend.generate(workflow, on_progress=progress)
            except Exception as primary_error:
                fallback_url = os.environ.get("ORBIT_COMFYUI_FALLBACK_URL", "").strip()
                if not fallback_url or fallback_url.rstrip("/") == backend.base_url:
                    raise
                self.store.emit(
                    task_id, "method_failed",
                    f"The primary image backend failed ({type(primary_error).__name__}: {primary_error}). I'm trying the configured fallback image backend."
                )
                fallback = ComfyUIImageBackend(base_url=fallback_url)
                try:
                    outputs = fallback.generate(workflow, on_progress=progress)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Primary image backend failed: {primary_error}; fallback also failed: {fallback_error}"
                    ) from fallback_error
            if not outputs:
                raise RuntimeError("The image backend completed without returning images.")
            GENERATED.mkdir(parents=True, exist_ok=True)
            for item in outputs:
                original = str(item.get("info", {}).get("filename", "image.png"))
                suffix = Path(original).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    suffix = ".png"
                name = uuid.uuid4().hex + suffix
                (GENERATED / name).write_bytes(item["content"])
                self.store.emit(task_id, "image", "Image generated.", url="/generated/" + name)
            self.store.update(task_id, "completed", result=f"Generated {len(outputs)} image(s).")
            self.store.emit(task_id, "completed", f"Generated {len(outputs)} image(s).")
        except Exception as exc:
            self.store.update(task_id, "failed", error=f"{type(exc).__name__}: {exc}")
            self.store.emit(task_id, "method_failed", f"Image generation failed: {type(exc).__name__}: {exc}. Check the configured image backend and workflow.")
        finally:
            with self._lock:
                self._threads = {t for t in self._threads if t.is_alive() and t is not threading.current_thread()}

    def _launch_chat(self, task_id, message):
        thread = threading.Thread(target=self._chat_worker, args=(task_id, message), daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()

    def _chat_worker(self, task_id, message):
        try:
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "Certainly. I'm considering your request.")
            # Run only narrowly matched, deterministic local utility intents.
            # All other messages are passed to the language model.
            agent = LocalAgent(
                workspace=Path(__file__).resolve().parent.parent,
                progress=lambda text, **details: self.store.emit(task_id, "status", text, **details),
                memory=self.memory,
            )
            answer = agent.try_execute(message)
            if answer is None:
                answer = self.runtime.reply(
                    message,
                    history=self.store.recent_history(),
                    owner_preferences=self.memory.prompt_context(),
                    on_progress=lambda event: self.store.emit(
                        task_id, event["kind"], event["message"],
                        **{key: value for key, value in event.items() if key not in ("kind", "message")}
                    ),
                )
            self.store.update(task_id, "completed", result=answer)
            self.store.emit(task_id, "message", answer)
        except ModelNotReady as exc:
            self.store.update(task_id, "needs_input", error=str(exc))
            self.store.emit(task_id, "method_failed",
                            f"The language model isn't ready: {exc}. Train a checkpoint, then restart Orbit.")
        except Exception as exc:
            self.store.update(task_id, "failed", error=f"{type(exc).__name__}: {exc}")
            self.store.emit(task_id, "method_failed",
                            f"I encountered an error: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._threads = {t for t in self._threads if t.is_alive() and t is not threading.current_thread()}

def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OrbitLocal/0.1"

        def log_message(self, fmt, *args):
            # Avoid dumping prompts or generated content into terminal logs.
            print("[Orbit HTTP] " + (fmt % args))

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                workflow_configured = bool(os.environ.get("ORBIT_IMAGE_WORKFLOW")) and Path(os.environ.get("ORBIT_IMAGE_WORKFLOW", "")).is_file()
                image_connected, image_error = False, None
                if workflow_configured:
                    try:
                        ComfyUIImageBackend(
                            base_url=os.environ.get("ORBIT_COMFYUI_URL", "http://127.0.0.1:8188"),
                            timeout=2,
                        ).health()
                        image_connected = True
                    except Exception as exc:
                        image_error = f"{type(exc).__name__}: {exc}"
                return self._json(200, {
                    "name": "Orbit", "status": "ready" if app.runtime.ready else "model_missing",
                    "device": app.runtime.device, "checkpoint": str(app.runtime.checkpoint_path),
                    "model_error": app.runtime.load_error,
                    "image_workflow_configured": workflow_configured,
                    "image_backend_connected": image_connected,
                    "image_backend_error": image_error,
                    "image_backend": os.environ.get("ORBIT_COMFYUI_URL", "http://127.0.0.1:8188")
                })
            if parsed.path == "/api/memory":
                return self._json(200, {"memories": app.memory.get_all()})
            if parsed.path == "/api/history":
                return self._json(200, {
                    "history": app.store.recent_history(limit=40),
                    "images": app.store.recent_images(limit=20),
                })
            if parsed.path == "/api/events":
                params = parse_qs(parsed.query)
                task_id = (params.get("task_id") or [""])[0]
                try:
                    after = int((params.get("after") or ["0"])[0])
                except ValueError:
                    return self._json(400, {"error": "after must be an integer"})
                if not task_id or not app.store.get(task_id):
                    return self._json(404, {"error": "Unknown task"})
                events = app.store.events(task_id, after)
                return self._json(200, {"events": events, "task": app.store.get(task_id)})
            if parsed.path.startswith("/generated/"):
                name = parsed.path.removeprefix("/generated/")
                target = (GENERATED / name).resolve()
                if GENERATED.resolve() not in target.parents or not target.is_file():
                    return self._json(404, {"error": "Not found"})
                content = target.read_bytes()
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
                return
            if parsed.path == "/" or parsed.path.startswith("/static/"):
                name = "index.html" if parsed.path == "/" else parsed.path.removeprefix("/static/")
                # Only serve files under the packaged web directory.
                target = (WEB / name).resolve()
                if WEB.resolve() not in target.parents or not target.is_file():
                    return self._json(404, {"error": "Not found"})
                content = target.read_bytes()
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
                return
            return self._json(404, {"error": "Not found"})

        def do_POST(self):
            route = urlparse(self.path).path
            if route not in ("/api/chat", "/api/memory", "/api/image"):
                return self._json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 32768:
                    return self._json(413, {"error": "Request body must be between 1 and 32768 bytes"})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    return self._json(400, {"error": "Request body must be a JSON object"})
                if route == "/api/memory":
                    key, value = payload.get("key"), payload.get("value")
                    app.memory.set(key, value)
                    return self._json(201, {"saved": True, "memories": app.memory.get_all()})
                if route == "/api/image":
                    prompt = payload.get("prompt")
                    negative_prompt = payload.get("negative_prompt", "")
                    seed = payload.get("seed")
                    if not isinstance(prompt, str) or not prompt.strip():
                        return self._json(400, {"error": "prompt must be a non-empty string"})
                    workflow_path = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
                    if not workflow_path or not Path(workflow_path).is_file():
                        return self._json(503, {"error": "Image workflow is not configured. Set ORBIT_IMAGE_WORKFLOW to a ComfyUI API workflow JSON file."})
                    if len(prompt) > 5000:
                        return self._json(400, {"error": "prompt must be no longer than 5000 characters"})
                    if not isinstance(negative_prompt, str) or len(negative_prompt) > 2000:
                        return self._json(400, {"error": "negative_prompt must be text up to 2000 characters"})
                    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**64 - 1):
                        return self._json(400, {"error": "seed must be an integer between 0 and 2^64-1"})
                    task_id = app.start_image(prompt.strip(), negative_prompt, seed)
                    return self._json(202, {"task_id": task_id, "status": "queued"})
                message = payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    return self._json(400, {"error": "message must be a non-empty string"})
                # Deterministic local tools remain available even before a language checkpoint is trained.
                task_id = app.start_chat(message.strip())
                return self._json(202, {"task_id": task_id, "status": "queued"})
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"error": "Invalid JSON"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/memory":
                return self._json(404, {"error": "Not found"})
            key = (parse_qs(parsed.query).get("key") or [""])[0]
            if not key:
                return self._json(400, {"error": "key query parameter is required"})
            return self._json(200, {"deleted": app.memory.delete(key), "memories": app.memory.get_all()})

    return Handler

def main():
    parser = argparse.ArgumentParser(description="Run the local Orbit chat interface.")
    parser.add_argument("--checkpoint", default="checkpoints/firstgpt.best.pt")
    parser.add_argument("--state", default="orbit_state.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    app = OrbitApplication(args.checkpoint, args.state)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Orbit is listening at http://{args.host}:{args.port}")
    if not app.runtime.ready:
        print(f"Model not loaded: {app.runtime.load_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Orbit.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
