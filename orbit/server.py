"""Local Orbit chat server with SQLite history and live task events.

Run: python -m orbit.server --checkpoint checkpoints/firstgpt.best.pt
This uses only Python's standard library plus the project's PyTorch dependency.
"""
from __future__ import annotations
import argparse
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import threading
import uuid

from .runtime import OrbitRuntime, ModelNotReady
from .tasks import TaskStore

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

class OrbitApplication:
    def __init__(self, checkpoint, state_path):
        self.runtime = OrbitRuntime(checkpoint)
        self.store = TaskStore(state_path)
        self.store.recover_interrupted()
        self._threads = set()
        self._lock = threading.Lock()

    def start_chat(self, message):
        task_id = self.store.create(message)
        thread = threading.Thread(target=self._chat_worker, args=(task_id, message), daemon=True)
        with self._lock:
            self._threads.add(thread)
        thread.start()
        return task_id

    def _chat_worker(self, task_id, message):
        try:
            self.store.update(task_id, "running")
            self.store.emit(task_id, "status", "Certainly. I'm considering your request.")
            answer = self.runtime.reply(
                message,
                on_progress=lambda event: self.store.emit(task_id, event["kind"], event["message"], **event)
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
                return self._json(200, {
                    "name": "Orbit", "status": "ready" if app.runtime.ready else "model_missing",
                    "device": app.runtime.device, "checkpoint": str(app.runtime.checkpoint_path),
                    "model_error": app.runtime.load_error
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
            if urlparse(self.path).path != "/api/chat":
                return self._json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 32768:
                    return self._json(413, {"error": "Request body must be between 1 and 32768 bytes"})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                message = payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    return self._json(400, {"error": "message must be a non-empty string"})
                if not app.runtime.ready:
                    return self._json(503, {"error": app.runtime.load_error, "hint": "Train a checkpoint and restart the server."})
                task_id = app.start_chat(message.strip())
                return self._json(202, {"task_id": task_id, "status": "queued"})
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"error": "Invalid JSON"})
            except Exception as exc:
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

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
