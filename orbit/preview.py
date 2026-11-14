"""Safe local development-preview manager for a selected Orbit workspace.

Only starts a project's declared package.json dev script (or a static preview).
The preview server binds to loopback and is never exposed on the LAN by default.
"""
from __future__ import annotations

import json
import os
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

    def _free_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _project_command(self, port):
        package = self.root / "package.json"
        if package.is_file():
            data = json.loads(package.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
            if isinstance(scripts, dict) and isinstance(scripts.get("dev"), str):
                manager = ("pnpm" if (self.root / "pnpm-lock.yaml").exists() else
                           "yarn" if (self.root / "yarn.lock").exists() else
                           "bun" if (self.root / "bun.lockb").exists() or (self.root / "bun.lock").exists() else "npm")
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
        if (self.root / "index.html").is_file():
            return [sys.executable, "-m", "orbit.preview_static", "--root", str(self.root), "--port", str(port)]
        raise RuntimeError("No runnable web project found. Orbit needs a package.json dev script or an index.html.")

    def status(self):
        running = self.process is not None and self.process.poll() is None
        return {"running": running, "url": "http://127.0.0.1:" + str(self.port) if running else None,
                "port": self.port if running else None, "command": self.command if running else None,
                "workspace": str(self.root)}

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return self.status()
        if not self.root.is_dir():
            raise RuntimeError("The selected project workspace does not exist.")
        port = self._free_port()
        command = self._project_command(port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        self.process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.DEVNULL,
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
            self.process.terminate()
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        old_port = self.port
        self.process = None
        self.port = None
        self.command = None
        return {"running": False, "stopped_port": old_port}
