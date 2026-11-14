"""Safe local development-preview manager for a selected Orbit workspace.

Only starts a project's declared package.json dev script (or a static preview).
The preview server binds to loopback and is never exposed on the LAN by default.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


class PreviewManager:
    def __init__(self, workspace: Path):
        self.root = Path(workspace).expanduser().resolve()
        self.process = None
        self.port = None
        self.command = None
        self.log_path = self.root / ".orbit_preview" / "server.log"
        self._log_handle = None
        self.workspace = self.root

    def _free_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def select_project(self, relative=None):
        if relative is None or not str(relative).strip():
            self.workspace = self.root
            return
        candidate = Path(str(relative))
        blocked = {".git", ".venv", "venv", "node_modules", ".orbit_preview"}
        if candidate.is_absolute() or ".." in candidate.parts or any(part in blocked or part.startswith(".env") for part in candidate.parts):
            raise PermissionError("Preview project path is not allowed.")
        target = (self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError("Preview project must remain inside the selected workspace.")
        current = self.root
        for part in candidate.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("Preview project paths cannot traverse symlinks.")
        if not target.is_dir():
            raise ValueError("The selected preview project folder does not exist.")
        self.workspace = target

    def _project_command(self, port):
        package = self.workspace / "package.json"
        if package.is_file():
            data = json.loads(package.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
            if isinstance(scripts, dict) and isinstance(scripts.get("dev"), str):
                manager = ("pnpm" if (self.workspace / "pnpm-lock.yaml").exists() else
                           "yarn" if (self.workspace / "yarn.lock").exists() else
                           "bun" if (self.workspace / "bun.lockb").exists() or (self.workspace / "bun.lock").exists() else "npm")
                executable = shutil.which(manager)
                if not executable:
                    raise RuntimeError(manager + " is required by this project but is not installed.")
                return [executable, "run", "dev", "--", "--host", "127.0.0.1",
                        "--port", str(port), "--strictPort"]
            if isinstance(scripts, dict) and isinstance(scripts.get("start"), str):
                manager = "npm"
                executable = shutil.which(manager)
                if not executable:
                    raise RuntimeError("npm is required by this project but is not installed.")
                return [executable, "run", "start", "--", "--host", "127.0.0.1",
                        "--port", str(port)]
            raise RuntimeError("This project has no package.json dev/start script. Add a development script before previewing.")
        if (self.workspace / "index.html").is_file():
            return [sys.executable, str(Path(__file__).with_name("preview_static.py")), "--root", str(self.workspace), "--port", str(port)]
        raise RuntimeError("No runnable web project found. Orbit needs a package.json dev script or an index.html.")

    def suggest_project(self, changes):
        """Suggest the deepest runnable project covering the largest share of changed files."""
        candidates = set()
        safe_files = []
        for item in changes if isinstance(changes, list) else []:
            relative = item.get("path") if isinstance(item, dict) else None
            if not isinstance(relative, str):
                continue
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                continue
            target = (self.root / candidate).resolve()
            if target != self.root and self.root not in target.parents:
                continue
            if any(part in {".git", ".venv", "venv", "node_modules", ".orbit_preview"} for part in candidate.parts):
                continue
            safe_files.append(target)
            parent = target.parent
            while parent == self.root or self.root in parent.parents:
                package = parent / "package.json"
                runnable = (parent / "index.html").is_file()
                if package.is_file():
                    try:
                        scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
                        runnable = runnable or (isinstance(scripts, dict) and
                            any(isinstance(scripts.get(key), str) for key in ("dev", "start")))
                    except (OSError, ValueError, AttributeError):
                        pass
                if runnable:
                    candidates.add(parent)
                if parent == self.root:
                    break
                parent = parent.parent
        if not candidates:
            return None
        ranked = []
        for directory in candidates:
            coverage = sum(1 for path in safe_files if path == directory or directory in path.parents)
            ranked.append((coverage, len(directory.parts), directory))
        selected = max(ranked, key=lambda row: (row[0], row[1]))[2]
        return selected.relative_to(self.root).as_posix() or "."

    def logs(self, limit=12000):
        try:
            raw = self.log_path.read_text(encoding="utf-8", errors="replace")[-max(1000, min(int(limit), 16000)):]
        except OSError:
            raw = ""
        # Avoid echoing common credential-shaped values from noisy build output.
        import re
        raw = re.sub(r"(?i)(api[_-]?key|access[_-]?token|secret|password)(\\s*[:=]\\s*)[^\\s,;]+",
                     r"\\1\\2[redacted]", raw)
        return {"available": bool(raw), "text": raw}

    def status(self):
        running = self.process is not None and self.process.poll() is None
        return {"running": running, "url": "http://127.0.0.1:" + str(self.port) if running else None,
                "port": self.port if running else None, "command": self.command if running else None,
                "workspace": str(self.workspace), "workspace_relative": str(self.workspace.relative_to(self.root)) if self.workspace != self.root else "."}

    def start(self, project_path=None):
        if self.process is not None and self.process.poll() is None:
            return self.status()
        if not self.root.is_dir():
            raise RuntimeError("The selected project workspace does not exist.")
        self.select_project(project_path)
        port = self._free_port()
        command = self._project_command(port)
        self.log_path = self.workspace / ".orbit_preview" / "server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        self.process = subprocess.Popen(command, cwd=self.workspace, stdin=subprocess.DEVNULL,
            stdout=self._log_handle, stderr=subprocess.STDOUT, shell=False,
            start_new_session=(os.name != "nt"))
        self.port, self.command = port, " ".join(command[:3])
        deadline = time.time() + 2.5
        while time.time() < deadline:
            if self.process.poll() is not None:
                self._log_handle.flush()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
                self.stop()
                raise RuntimeError("Preview server exited during startup. " + tail)
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return self.status()
            except OSError:
                time.sleep(0.1)
        return self.status()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=4)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    if os.name != "nt":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait(timeout=2)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        old_port = self.port
        self.process = None
        self.port = None
        self.command = None
        return {"running": False, "stopped_port": old_port}
