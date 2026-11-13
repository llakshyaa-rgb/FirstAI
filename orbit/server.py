"""Local Orbit chat server with SQLite history and live task events.

Run: python -m orbit.server --checkpoint checkpoints/firstgpt.best.pt
This uses only Python's standard library plus the project's PyTorch dependency.
"""
from __future__ import annotations
import argparse
import json
import mimetypes
from importlib.util import find_spec
import os
import re
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from .runtime import OrbitRuntime, ModelNotReady
from .tasks import TaskStore
from .memory import MemoryStore
from .images import ComfyUIImageBackend
from .image_workflow import load_workflow_template
from .agent import LocalAgent
from .media import parse_multipart, save_upload, list_uploads, get_upload, delete_upload, search_web_images, UPLOAD_DIR, MAX_UPLOAD_BYTES
from .vision import describe_image, unload_ollama_vision_model
from .media_processing import process_upload
from .planner import execute_tool_plan
from .agents import AgentTeam
from .intent import classify_request
from .browser import BrowserTools
from .design_studio import create_artifact, SUPPORTED, THEMES

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
GENERATED = ROOT / "generated"

class OrbitApplication:
    def __init__(self, checkpoint, state_path):
        self.runtime = OrbitRuntime(checkpoint)
        self.browser = BrowserTools()
        self._diffusion_backend = None
        self._diffusion_backend_lock = threading.Lock()
        self.store = TaskStore(state_path)
        self.memory = MemoryStore(state_path)
        self._threads = set()
        self._lock = threading.Lock()
        worker_count = max(1, min(4, int(os.environ.get('ORBIT_TASK_WORKERS', '3'))))
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='orbit-task')
        # Resume interrupted chat-generation tasks after the checkpoint is loaded.
        for task_id in self.store.recover_interrupted():
            task = self.store.get(task_id)
            if task and task["goal"].startswith("Generate image: "):
                self.store.update(task_id, "needs_input", error="Image generation was interrupted by a server restart. Resubmit the image request.")
                self.store.emit(task_id, "method_failed", "The image job was interrupted by a restart. Its prompt is preserved; please resubmit it.")
            elif task and task["goal"].startswith("Design ") and self.runtime.ready:
                requests = [event for event in self.store.events(task_id) if event["kind"] == "design_request"]
                if requests:
                    details = requests[-1]["details"]
                    self.store.update(task_id, "queued")
                    self._launch_design(task_id, details["title"], details["brief"], details["format"], details["theme"])
                else:
                    self.store.update(task_id, "needs_input", error="Design request details are missing. Please submit it again.")
            elif task and self.runtime.ready:
                self.store.update(task_id, "queued")
                attachment_events = [event for event in self.store.events(task_id) if event["kind"] == "attachments"]
                attachments = attachment_events[-1]["details"].get("file_ids", []) if attachment_events else []
                self._launch_chat(task_id, task["goal"], attachments)

    def _submit(self, worker, *args):
        """Queue durable tasks on a bounded pool instead of spawning unbounded threads."""
        future = self._executor.submit(worker, *args)
        with self._lock:
            self._threads.add(future)
        future.add_done_callback(lambda done: self._discard_future(done))
        return future

    def _discard_future(self, future):
        with self._lock:
            self._threads.discard(future)

    def start_chat(self, message, attachments=None):
        route = classify_request(message)
        # Artifacts and generated images are still submitted from the same chat box.
        if route["kind"] == "image_generation":
            backend_kind = os.environ.get("ORBIT_IMAGE_BACKEND", "comfyui").strip().lower()
            if backend_kind == "diffusers" and find_spec("diffusers") is None:
                raise RuntimeError("Diffusers is not installed. Install requirements-image.txt or configure ComfyUI.")
            if backend_kind == "comfyui":
                workflow_path = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
                if not workflow_path or not Path(workflow_path).is_file():
                    raise RuntimeError("Image generation is not configured. Configure ComfyUI or enable the Diffusers backend.")
            elif backend_kind != "diffusers":
                raise ValueError("ORBIT_IMAGE_BACKEND must be comfyui or diffusers.")
            return self.start_image(message)
        task_id = self.store.create(message)
        if attachments:
            self.store.emit(task_id, "attachments", "File attachments linked to this task.", file_ids=list(attachments[:5]))
        self._submit(self._chat_worker, task_id, message, attachments or [])
        return task_id

    def retry_task(self, task_id):
        task = self.store.get(task_id)
        if not task:
            raise KeyError(task_id)
        if task["status"] not in {"failed", "interrupted", "needs_input"}:
            raise ValueError("Only failed, interrupted or waiting tasks can be retried.")
        goal = task["goal"]
        if goal.startswith("Generate image: "):
            prompt = goal.removeprefix("Generate image: ")
            self.store.update(task_id, "queued", result=None, error=None)
            self.store.emit(task_id, "queued", "Image task queued for another attempt.")
            self._submit(self._image_worker, task_id, prompt, "", None)
            return task_id
        if goal.startswith("Design "):
            requests = [event for event in self.store.events(task_id) if event["kind"] == "design_request"]
            if requests:
                details = requests[-1]["details"]
                if not self.runtime.ready:
                    raise RuntimeError("A trained FirstGPT checkpoint is required before retrying this design.")
                self.store.update(task_id, "queued", result=None, error=None)
                self.store.emit(task_id, "queued", "Design task queued for another attempt.")
                self._launch_design(task_id, details["title"], details["brief"], details["format"], details["theme"])
                return task_id
        if not self.runtime.ready:
            raise RuntimeError("A trained FirstGPT checkpoint is required before retrying this chat task.")
        attachment_events = [event for event in self.store.events(task_id) if event["kind"] == "attachments"]
        attachments = attachment_events[-1]["details"].get("file_ids", []) if attachment_events else []
        self.store.update(task_id, "queued", result=None, error=None)
        self.store.emit(task_id, "queued", "Chat task queued for another attempt.")
        self._submit(self._chat_worker, task_id, goal, attachments)
        return task_id

    def start_design(self, title, brief, file_format="pdf", theme="midnight"):
        task_id = self.store.create("Design " + file_format.upper() + ": " + title)
        self.store.emit(task_id, "design_request", "Design request saved for recovery.",
                        title=title, brief=brief, format=file_format, theme=theme)
        self._launch_design(task_id, title, brief, file_format, theme)
        return task_id

    def _launch_design(self, task_id, title, brief, file_format, theme):
        self._submit(self._design_worker, task_id, title, brief, file_format, theme)
    def _design_worker(self, task_id, title, brief, file_format, theme):
        try:
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "Orbit is planning the structure and writing the content.")
            history = self.store.recent_history(limit=4)
            preferences = self.memory.get_all()
            preference_text = "; ".join(str(item.get("key", "")) + ": " + str(item.get("value", "")) for item in preferences[:8])
            prompt = (
                "Create the complete content for a polished " + file_format.upper() + " deliverable. "
                "Title: " + title[:200] + ". Design brief: " + brief[:6000] + ". "
                "Return only finished document content in clean Markdown. Use a strong title hierarchy, useful sections, specific details, and tables or bullets where useful. "
                "Do not describe what you would do. Do not include an AI introduction. Do not invent sources, statistics or citations; label assumptions clearly. "
                "For presentations, use slide headings and concise slide-ready points. For spreadsheets, use tables with headers and rows."
            )
            self.store.emit(task_id, "status", "Drafting the deliverable with the language model.")
            content = self.runtime.reply(
                prompt, history=history, owner_preferences=preference_text,
                max_new_tokens=max(self.runtime.max_new_tokens, min(512, int(os.environ.get("ORBIT_DESIGN_MAX_TOKENS", "384")))),
                on_progress=lambda event: self.store.emit(task_id, event.get("kind", "status"), event.get("message", ""),
                    **{key: value for key, value in event.items() if key not in ("kind", "message")})
            )
            if not content or not content.strip():
                raise RuntimeError("The language model returned an empty design draft.")
            self.store.emit(task_id, "status", "Formatting and exporting the finished file.")
            artifact = create_artifact(title, content, file_format, theme_name=theme)
            self.store.emit(task_id, "artifact", "Your design is ready.", artifact=artifact)
            self.store.update(task_id, "completed", result=json.dumps(artifact, ensure_ascii=False))
            self.store.emit(task_id, "completed", "Created " + artifact["name"])
        except Exception as exc:
            self.store.update(task_id, "failed", error=type(exc).__name__ + ": " + str(exc))
            self.store.emit(task_id, "method_failed", "Design creation failed: " + type(exc).__name__ + ": " + str(exc))
    def start_image(self, prompt, negative_prompt="", seed=None):
        task_id = self.store.create("Generate image: " + prompt)
        self._submit(self._image_worker, task_id, prompt, negative_prompt, seed)
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
                with self._diffusion_backend_lock:
                    if self._diffusion_backend is None:
                        self._diffusion_backend = DiffusionImageBackend()
                    backend = self._diffusion_backend
                unload_ollama_vision_model()
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
    def _launch_chat(self, task_id, message, attachments=None):
        self._submit(self._chat_worker, task_id, message, attachments or [])

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
                workspace=Path(os.environ.get("ORBIT_WORKSPACE_ROOT", str(Path(__file__).resolve().parent.parent))).expanduser().resolve(),
                progress=emit_progress,
                memory=self.memory,
                browser=self.browser,
            )
            # One automatic capability router serves the single conversational interface.
            route = classify_request(message)
            specialist_results = AgentTeam(
                agent.registry, progress=emit_progress
            ).run(message, attachments=attachments, capabilities=route.get("capabilities", []))
            for item in specialist_results:
                if item.get("agent") == "image_researcher" and item.get("ok"):
                    image_results = item.get("result") or []
                    self.store.emit(task_id, "image_results", "Found image references.", results=image_results)
                    context_parts.append("Image-search results (titles and source links): " + json.dumps(
                        [{"title": x.get("title"), "source": x.get("source"), "page_url": x.get("page_url")}
                         for x in image_results], ensure_ascii=False
                    )[:6000])
                    continue
                if item.get("agent") != "media":
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
            findings = [item for item in specialist_results if item.get("ok")]
            if findings:
                message += "\n\nSpecialist findings (tool output; evaluate relevance and verify claims):\n" + json.dumps(
                    findings, ensure_ascii=False, default=str
                )[:16000]
            failed_agents = [item for item in specialist_results if not item.get("ok")]
            if failed_agents:
                message += "\n\nSpecialist failures (do not claim these succeeded):\n" + json.dumps(
                    failed_agents, ensure_ascii=False, default=str
                )[:4000]

            if route.get("kind") == "artifact":
                self.store.emit(task_id, "status", "The coordinator is preparing the requested deliverable.")
                brief = message[:9000]
                prompt = (
                    "Create a complete, polished " + str(route.get("format") or "pdf").upper() +
                    " deliverable. Request: " + brief + ". Return only finished Markdown content. "
                    "Use clear hierarchy, useful sections, concrete details, tables or bullets where useful. "
                    "For presentations, structure content as slide headings and concise slide-ready points. "
                    "For spreadsheets, use tables with headers and rows. Do not invent facts or sources; "
                    "label assumptions. Do not explain what you would do."
                )
                draft = self.runtime.reply(
                    prompt, history=self.store.recent_history(limit=4),
                    owner_preferences=self.memory.prompt_context(),
                    max_new_tokens=max(self.runtime.max_new_tokens, min(512, int(os.environ.get("ORBIT_DESIGN_MAX_TOKENS", "384")))),
                    on_progress=lambda event: emit_progress(event["message"], kind=event["kind"],
                        **{key: value for key, value in event.items() if key not in ("kind", "message")}),
                )
                if not draft or not draft.strip():
                    raise RuntimeError("The model returned an empty deliverable draft.")
                artifact = create_artifact(
                    route.get("title") or "Orbit deliverable", draft,
                    route.get("format") or "pdf", theme_name="midnight",
                )
                self.store.emit(task_id, "artifact", "Your deliverable is ready.", artifact=artifact)
                self.store.update(task_id, "completed", result=json.dumps(artifact, ensure_ascii=False))
                self.store.emit(task_id, "completed", "Created " + artifact["name"])
                return

            answer = agent.try_execute(message)
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
                    image_connected = find_spec("diffusers") is not None
                    if not image_connected:
                        image_error = "Diffusers is not installed. Run pip install -r requirements-image.txt."
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
                    "workspace_root": str(Path(os.environ.get("ORBIT_WORKSPACE_ROOT", str(Path(__file__).resolve().parent.parent))).expanduser().resolve()),
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
                    "task_workers": max(1, min(4, int(os.environ.get("ORBIT_TASK_WORKERS", "3")))),
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
                self.send_header("Cache-Control", "private, no-store")
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
            if route not in ("/api/chat", "/api/memory", "/api/image", "/api/upload", "/api/vision", "/api/design", "/api/tasks/retry"):
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
                if route == "/api/tasks/retry":
                    task_id = payload.get("task_id")
                    if not isinstance(task_id, str) or not task_id:
                        return self._json(400, {"error": "task_id is required"})
                    task_id = app.retry_task(task_id)
                    return self._json(202, {"task_id": task_id, "status": "queued"})
                if route == "/api/design":
                    title = payload.get("title", "Orbit design")
                    brief = payload.get("brief", "")
                    file_format = payload.get("format", "pdf")
                    theme = payload.get("theme", "midnight")
                    if not all(isinstance(value, str) for value in (title, brief, file_format, theme)):
                        return self._json(400, {"error": "title, brief, format and theme must be text"})
                    if not brief.strip() or len(brief) > 6000:
                        return self._json(400, {"error": "Describe what to design in 1-6000 characters."})
                    file_format = file_format.lower().lstrip(".")
                    theme = theme.lower()
                    if file_format not in SUPPORTED:
                        return self._json(400, {"error": "Unsupported export format. Choose: " + ", ".join(sorted(SUPPORTED))})
                    if theme not in THEMES:
                        return self._json(400, {"error": "Unknown design theme. Choose: " + ", ".join(sorted(THEMES))})
                    if not title.strip():
                        return self._json(400, {"error": "A design title is required."})
                    if not app.runtime.ready:
                        return self._json(503, {"error": "The design studio needs a trained FirstGPT checkpoint. Load one before requesting AI-written deliverables."})
                    task_id = app.start_design(title, brief, file_format, theme)
                    return self._json(202, {"task_id": task_id, "status": "queued"})
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
                    backend_kind = os.environ.get("ORBIT_IMAGE_BACKEND", "comfyui").strip().lower()
                    if backend_kind == "comfyui":
                        workflow_path = os.environ.get("ORBIT_IMAGE_WORKFLOW", "").strip()
                        if not workflow_path or not Path(workflow_path).is_file():
                            return self._json(503, {"error": "ComfyUI workflow is not configured. Set ORBIT_IMAGE_WORKFLOW, or select Diffusers in Colab."})
                    elif backend_kind == "diffusers":
                        if find_spec("diffusers") is None:
                            return self._json(503, {"error": "Diffusers is not installed. Run pip install -r requirements-image.txt."})
                    else:
                        return self._json(400, {"error": "ORBIT_IMAGE_BACKEND must be comfyui or diffusers."})
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
            if parsed.path == "/api/uploads":
                file_id = (parse_qs(parsed.query).get("id") or [""])[0]
                if not file_id:
                    return self._json(400, {"error": "id query parameter is required"})
                try:
                    delete_upload(file_id)
                    return self._json(200, {"deleted": True, "files": list_uploads()})
                except FileNotFoundError:
                    return self._json(404, {"error": "Uploaded file not found"})
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
        if app._diffusion_backend is not None:
            app._diffusion_backend.unload()

if __name__ == "__main__":
    main()
