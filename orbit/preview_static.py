"""Loopback-only static preview that refuses hidden and sensitive project files."""
from __future__ import annotations
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import os

BLOCKED = {".git", ".venv", "venv", "__pycache__", "node_modules",
           ".orbit_backups", ".orbit_preview", "dist", "build"}

def handler_for(root):
    root = Path(root).resolve()
    class SafeHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)
        def translate_path(self, path):
            raw = unquote(urlsplit(path).path).lstrip("/")
            parts = [p for p in raw.split("/") if p not in ("", ".")]
            if any(p.startswith(".") or p in BLOCKED for p in parts):
                return str(root / "__orbit_forbidden__")
            target = root.joinpath(*parts).resolve()
            if target != root and root not in target.parents:
                return str(root / "__orbit_forbidden__")
            if any(part.is_symlink() for part in [root.joinpath(*parts[:i+1]) for i in range(len(parts))]):
                return str(root / "__orbit_forbidden__")
            return str(target)
        def log_message(self, fmt, *args):
            pass
    return SafeHandler

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--port",type=int,required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if not root.is_dir(): raise SystemExit("Preview root does not exist")
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler_for(root))
    try: server.serve_forever()
    finally: server.server_close()

if __name__=="__main__": main()
