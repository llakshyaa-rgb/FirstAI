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
from .media import parse_multipart, save_upload, list_uploads, get_upload, search_web_images, UPLOAD_DIR, MAX_UPLOAD_BYTES
from .vision import describe_image
from .media_processing import process_upload
from .planner import execute_tool_plan
from .agents import AgentTeam
from .browser import BrowserTools

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
GENERATED = ROOT / "generated"

class OrbitApplication:
    def __init__(self, checkpoint, state_path):
        self.runtime = OrbitRuntime(checkpoint)
        self.browser = BrowserTools()
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
                attachment_events = [event for event in self.store.events(task_id) if event["kind"] == "attachments"]
                attachments = attachment_events[-1]["details"].get("file_ids", []) if attachment_events else []
                self._launch_chat(task_id, task["goal"], attachments)

    def start_chat(self, message, attachments=None):
        task_id = self.store.create(message)
        if attachments:
            self.store.emit(task_id, "attachments", "File attachments linked to this task.", file_ids=list(attachments[:5]))
        thread = threading.Thread(target=self._chat_worker, args=(task_id, message, attachments or []), daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()
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
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "The image backend is preparing the requested artwork.")
            backend_kind = os.environ.get("ORBIT_IMAGE_BACKEND", "comfyui").strip().lower()
            progress = lambda event: self.store.emit(
                task_id, event["kind"], event["message"],
                **{key: value for key, value in event.items() if key not in ("kind", "message")}
            )
            if backend_kind == "diffusers":
                from .diffusion_backend import DiffusionImageBackend
                backend = DiffusionImageBackend()
                with self.runtime.offload_for_external_gpu_work():
                    outputs = backend.generate(
                        prompt, negative_prompt=negative_prompt, seed=seed,
                        width=int(os.environ.get("ORBIT_IMAGE_WIDTH", "512")),
                        height=int(os.environ.get("ORBIT_IMAGE_HEIGHT", "512")),
                        steps=int(os.environ.get("ORBIT_IMAGE_STEPS", "20")),
                        guidance=float(os.environ.get("ORBIT_IMAGE_CFG", "7.0")),
                        on_progress=progress,
                    )
            elif backend_kind == "comfyui":
                workflow_path = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
                if not workflow_path or not Path(workflow_path).is_file():
                    raise RuntimeError("ComfyUI workflow is not configured. Set ORBIT_IMAGE_WORKFLOW, or select ORBIT_IMAGE_BACKEND=diffusers and install requirements-image.txt.")
                workflow = load_workflow_template(
                    workflow_path, prompt, negative_prompt=negative_prompt, seed=seed,
                    checkpoint=os.environ.get("ORBIT_IMAGE_CHECKPOINT", "model.safetensors"),
                    width=int(os.environ.get("ORBIT_IMAGE_WIDTH", "768")),
                    height=int(os.environ.get("ORBIT_IMAGE_HEIGHT", "768")),
                    steps=int(os.environ.get("ORBIT_IMAGE_STEPS", "24")),
                    cfg=float(os.environ.get("ORBIT_IMAGE_CFG", "7.0")),
                )
                backend = ComfyUIImageBackend(base_url=os.environ.get("ORBIT_COMFYUI_URL", "http://127.0.0.1:8188"))
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
            else:
                raise ValueError("ORBIT_IMAGE_BACKEND must be comfyui or diffusers.")
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

    def _launch_chat(self, task_id, message, attachments=None):
        thread = threading.Thread(target=self._chat_worker, args=(task_id, message, attachments or []), daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()

    def _chat_worker(self, task_id, message, attachments=None):
        execution_failures = []

        def emit_progress(message, kind="status", **details):
            if kind == "method_failed":
                execution_failures.append(str(message))
            self.store.emit(task_id, kind, message, **details)

        try:
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "Certainly. I'm considering your request.")
            context_parts = []
            specialist_results = []
            agent = LocalAgent(
                workspace=Path(__file__).resolve().parent.parent,
                progress=emit_progress,
                memory=self.memory,
                browser=self.browser,
            )
            # Attachments are delegated to a dedicated media specialist. Other
            # relevant specialists can research or inspect the workspace in parallel.
            if attachments:
                team = AgentTeam(agent.registry, progress=emit_progress)
                specialist_results = team.run(message, attachments=attachments)
                for item in specialist_results:
                    if item.get("agent") != "media" or not item.get("ok"):
                        continue
                    for media_item in item.get("result", []):
                        if media_item.get("content"):
                            context_parts.append(
                                media_item.get("kind", "file").capitalize() + " " +
                                media_item.get("name", "attachment") + " analysis:\n" +
                                media_item["content"]
                            )
                        elif media_item.get("fallback_text"):
                            context_parts.append(
                                "Readable extracted text from attachment:\n" + media_item["fallback_text"]
                            )
                        elif not media_item.get("ok"):
                            emit_progress(
                                "An attachment could not be fully interpreted (" +
                                media_item.get("error", "unknown error") +
                                "). I'll continue with any readable fallback content.",
                                kind="method_failed",
                            )
                if context_parts:
                    message += "\n\nAttached material:\n" + "\n\n".join(context_parts)
                findings = [item for item in specialist_results if item.get("agent") != "media"]
                if findings:
                    message += "\n\nOther specialist findings:\n" + json.dumps(findings, ensure_ascii=False, default=str)[:12000]

            answer = agent.try_execute(message)
            if answer is None and not specialist_results:
                team = AgentTeam(
                    agent.registry,
                    progress=emit_progress,
                )
                specialist_results = team.run(message)
                if specialist_results:
                    message = message + "\n\nSpecialist-agent findings (tool results; verify before relying on them):\n" + json.dumps(specialist_results, ensure_ascii=False, default=str)[:16000]
            if answer is None and not specialist_results and self.runtime.ready and any(
                word in message.casefold() for word in
                ("search", "research", "look up", "calculate", "compute", "list files",
                 "read file", "inspect", "find in", "compare", "analyse", "analyze", "check project")
            ):
                plan_text = self.runtime.plan_tool_call(message, agent.registry.describe())
                planned = execute_tool_plan(
                    plan_text, agent.registry,
                    progress=lambda text: emit_progress(text),
                )
                if planned:
                    message = message + "\n\nSafe tool execution results (treat as factual tool output):\n" + json.dumps(planned, ensure_ascii=False)[:12000]
            if answer is None:
                if execution_failures:
                    message += "\n\nMethods already attempted and their failures (do not claim they succeeded; try an available alternative):\n" + "\n".join("- " + item for item in execution_failures[-8:])
                answer = self.runtime.reply(
                    message,
                    history=self.store.recent_history(),
                    owner_preferences=self.memory.prompt_context(),
                    on_progress=lambda event: emit_progress(
                        event["message"], kind=event["kind"],
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
            # Do not log query strings, filenames, or potentially sensitive upload/search details.
            print("[Orbit HTTP] request handled")

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
                image_backend = os.environ.get("ORBIT_IMAGE_BACKEND", "comfyui").strip().lower()
                workflow_configured = bool(os.environ.get("ORBIT_IMAGE_WORKFLOW")) and Path(os.environ.get("ORBIT_IMAGE_WORKFLOW", "")).is_file()
                image_connected, image_error = False, None
                if image_backend == "diffusers":
                    image_connected = True
                elif workflow_configured:
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
                    "image_backend": image_backend,
                    "image_model": os.environ.get("ORBIT_IMAGE_MODEL", "Lykon/dreamshaper-8"),
                    "image_search_configured": bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
                    "vision_configured": bool(os.environ.get("ORBIT_VISION_MODEL", "").strip()),
                    "vision_provider": os.environ.get("ORBIT_VISION_PROVIDER", "ollama"),
                    "agent_workers": max(1, min(4, int(os.environ.get("ORBIT_AGENT_WORKERS", "3")))),
                    "gpu_memory_fraction": app.runtime.gpu_memory_fraction,
                    "cpu_threads": app.runtime.cpu_threads,
                    "max_new_tokens": app.runtime.max_new_tokens,
                    "gpu_memory_policy": "per-process PyTorch allocator ceiling; not a target utilization percentage"
                })
            if parsed.path == "/api/memory":
                return self._json(200, {"memories": app.memory.get_all()})
            if parsed.path == "/api/uploads":
                return self._json(200, {"files": list_uploads()})
            if parsed.path == "/api/image-search":
                params = parse_qs(parsed.query)
                query = (params.get("q") or [""])[0]
                safesearch = (params.get("safe") or ["off"])[0]
                try:
                    return self._json(200, {"query": query, "results": search_web_images(query, safesearch=safesearch), "may_include_mature": safesearch == "off"})
                except RuntimeError as exc:
                    return self._json(503, {"error": str(exc)})
                except Exception as exc:
                    return self._json(502, {"error": f"Image search failed: {type(exc).__name__}: {exc}"})
            if parsed.path.startswith("/uploads/"):
                name = parsed.path.removeprefix("/uploads/")
                target = (UPLOAD_DIR / name).resolve()
                if UPLOAD_DIR.resolve() not in target.parents or not target.is_file():
                    return self._json(404, {"error": "Not found"})
                content = target.read_bytes()
                ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_response(200)
                safe_inline = target.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".heic", ".heif", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".aif", ".opus", ".amr", ".wma", ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".3gp", ".mpeg", ".mpg", ".wmv", ".flv", ".ts"}
                self.send_header("Content-Type", ctype if safe_inline else "application/octet-stream")
                self.send_header("Content-Disposition", "inline" if safe_inline else "attachment; filename=\"download\"")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
                self.send_header("Cache-Control", "private, max-age=3600")
                self.end_headers()
                self.wfile.write(content)
                return
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
            if route not in ("/api/chat", "/api/memory", "/api/image", "/api/upload", "/api/vision"):
                return self._json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if route == "/api/upload":
                    if length < 1 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
                        return self._json(413, {"error": "Upload request must be 1 byte to 21 MB."})
                    files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
                    saved = [save_upload(name, data, content_type) for name, data, content_type in files]
                    return self._json(201, {"files": saved})
                if length < 1 or length > 32768:
                    return self._json(413, {"error": "Request body must be between 1 and 32768 bytes"})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    return self._json(400, {"error": "Request body must be a JSON object"})
                if route == "/api/vision":
                    file_id = payload.get("file_id")
                    prompt = payload.get("prompt", "Describe this image in detail.")
                    if not isinstance(file_id, str) or not isinstance(prompt, str):
                        return self._json(400, {"error": "file_id and prompt must be text"})
                    processed = process_upload(file_id, prompt)
                    return self._json(200, {"description": processed["content"], "kind": processed["kind"], "name": processed["name"]})
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
                attachments = payload.get("attachments", [])
                if not isinstance(message, str) or not message.strip():
                    return self._json(400, {"error": "message must be a non-empty string"})
                if not isinstance(attachments, list) or len(attachments) > 5 or any(not isinstance(x, str) for x in attachments):
                    return self._json(400, {"error": "attachments must be a list of up to five uploaded file IDs"})
                # Deterministic local tools remain available even before a language checkpoint is trained.
                task_id = app.start_chat(message.strip(), attachments)
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
        app.browser.close()

if __name__ == "__main__":
    main()
