"""Narrow, workspace-confined tools for Orbit's private local assistant.

No arbitrary shell, eval, network access, or unrestricted filesystem access.
"""
from __future__ import annotations
import ast
import math
import operator
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

_BLOCKED_PARTS = {".git", ".venv", "venv", "__pycache__", ".env", ".secrets", ".orbit_backups"}
_MAX_TEXT_BYTES = 256_000
_MAX_WRITE_BYTES = 1_000_000
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str):
    """Evaluate a restricted arithmetic expression without eval or Python calls."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 300:
        raise ValueError("Enter an arithmetic expression of at most 300 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("That is not a valid arithmetic expression.") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 64:
        raise ValueError("Expression is too complex.")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large.")
            value = _BINOPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = _UNARY[type(node.op)](visit(node.operand))
        else:
            raise ValueError("Only numbers and basic arithmetic operators are allowed.")
        if isinstance(value, complex) or not math.isfinite(float(value)) or abs(value) > 1e100:
            raise ValueError("Result is outside the supported numeric range.")
        return value
    try:
        return visit(tree)
    except (ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Cannot calculate that expression: {exc}") from exc


class WorkspaceTools:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parent.parent).resolve()
        self.backup_root = self.root / ".orbit_backups"

    def _path(self, relative):
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("A relative workspace path is required.")
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError("Absolute paths are not allowed; use a path inside the workspace.")
        if any(part in _BLOCKED_PARTS or part.startswith(".env") for part in candidate.parts):
            raise PermissionError("That path is private or excluded from workspace tools.")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionError("That path escapes the Orbit workspace.")
        resolved_parts = resolved.relative_to(self.root).parts
        if any(part in _BLOCKED_PARTS or part.startswith(".env") for part in resolved_parts):
            raise PermissionError("That path resolves to private or excluded workspace data.")
        return resolved

    def list_files(self, relative="."):
        target = self._path(relative)
        if not target.exists() or not target.is_dir():
            raise NotADirectoryError(relative)
        items = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.is_symlink() or child.name in _BLOCKED_PARTS or child.name.startswith(".env"):
                continue
            items.append({
                "name": child.name + ("/" if child.is_dir() else ""),
                "type": "directory" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
            if len(items) >= 200:
                break
        return items

    def read_file(self, relative):
        target = self._path(relative)
        if not target.is_file():
            raise FileNotFoundError(relative)
        size = target.stat().st_size
        if size > _MAX_TEXT_BYTES:
            raise ValueError(f"File is too large to display ({size:,} bytes; limit {_MAX_TEXT_BYTES:,}).")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Only UTF-8 text files can be read by this tool.") from exc

    def search_files(self, query, relative="."):
        if not isinstance(query, str) or not query.strip() or len(query) > 200:
            raise ValueError("Search text must be 1-200 characters.")
        base = self._path(relative)
        if not base.is_dir():
            raise NotADirectoryError(relative)
        matches = []
        scanned = 0
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file() or any(part in _BLOCKED_PARTS or part.startswith(".env") for part in path.relative_to(self.root).parts):
                continue
            scanned += 1
            if scanned > 1500:
                break
            try:
                if path.stat().st_size > _MAX_TEXT_BYTES:
                    continue
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append({"path": path.relative_to(self.root).as_posix(),
                                        "line": line_no, "text": line[:500]})
                        if len(matches) >= 100:
                            return matches
            except (OSError, UnicodeDecodeError):
                continue
        return matches

    def inspect_project(self, relative=".", max_files=60):
        """Return a bounded source map: project tree, imports, definitions and entrypoints."""
        base = self._path(relative)
        if not base.is_dir():
            raise NotADirectoryError(relative)
        ignored = _BLOCKED_PARTS | {".pytest_cache", "node_modules", "dist", "build", ".next"}
        files = []
        source_map = []
        candidates = []
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            rel_parts = path.relative_to(self.root).parts
            if any(part in ignored or part.startswith(".env") for part in rel_parts):
                continue
            if path.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".json", ".toml", ".yaml", ".yml"}:
                continue
            candidates.append(path)
            if len(candidates) >= max(1, min(int(max_files), 100)):
                break
        for path in candidates:
            rel = path.relative_to(self.root).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size})
            if path.stat().st_size > 160_000:
                continue
            try:
                source = path.read_text(encoding="utf-8")
                imports, definitions = [], []
                if path.suffix.lower() == ".py":
                    tree = ast.parse(source, filename=rel)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            imports.append(ast.get_source_segment(source, node) or "")
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            definitions.append({
                                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                                "name": node.name, "line": node.lineno,
                            })
                else:
                    for line_no, line in enumerate(source.splitlines(), 1):
                        if re.match(r"\s*(?:import |from |const .*require\(|const .*from |export .*from )", line):
                            imports.append(line.strip()[:240])
                        match = re.match(r"\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)", line)
                        if match:
                            definitions.append({"kind": "symbol", "name": match.group(1), "line": line_no})
                source_map.append({
                    "path": rel, "imports": imports[:30], "definitions": definitions[:80],
                })
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                source_map.append({"path": rel, "parse_error": type(exc).__name__})
        return {
            "root": str(base.relative_to(self.root)) if base != self.root else ".",
            "files_scanned": len(files),
            "files": files,
            "python_source_map": source_map,
            "limits": "Bounded source map; excluded secrets, environments, dependencies and generated build folders.",
        }

    def create_file(self, relative, content):
        target = self._path(relative)
        if not isinstance(content, str) or len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            raise ValueError("File content must be text no larger than 1 MB.")
        if target.exists():
            raise FileExistsError(f"{relative} already exists; refusing to overwrite it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(target, content)
        return {"path": target.relative_to(self.root).as_posix(), "bytes": target.stat().st_size}

    def update_file(self, relative, content):
        target = self._path(relative)
        if not target.is_file():
            raise FileNotFoundError(relative)
        if not isinstance(content, str) or len(content.encode("utf-8")) > _MAX_WRITE_BYTES:
            raise ValueError("File content must be text no larger than 1 MB.")
        if target.stat().st_size > _MAX_WRITE_BYTES:
            raise ValueError("Refusing to update a file larger than 1 MB; no backup would be safe.")
        self.backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.backup_root / (stamp + "-" + target.name + ".bak")
        shutil.copy2(target, backup)
        self._atomic_write(target, content)
        return {"path": target.relative_to(self.root).as_posix(),
                "bytes": target.stat().st_size,
                "backup": backup.relative_to(self.root).as_posix()}

    @staticmethod
    def _atomic_write(target, content):
        fd, temp_name = tempfile.mkstemp(prefix=".orbit-", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def run_project_tests(self, timeout=180):
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        try:
            result = subprocess.run(command, cwd=self.root, capture_output=True,
                                    text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Project tests exceeded {timeout} seconds.") from exc
        output = (result.stdout + "\n" + result.stderr).strip()
        return {"returncode": result.returncode, "passed": result.returncode == 0,
                "command": "python -m unittest discover -s tests -v",
                "output": output[-16000:]}

    def compile_project(self, timeout=120):
        ignored = _BLOCKED_PARTS | {".pytest_cache", ".mypy_cache", ".ruff_cache",
                                    "node_modules", "dist", "build", ".next"}
        sources = []
        for path in self.root.rglob("*.py"):
            if path.is_symlink() or any(part in ignored or part.startswith(".env")
                                        for part in path.relative_to(self.root).parts):
                continue
            sources.append(path.relative_to(self.root).as_posix())
            if len(sources) >= 300:
                break
        if not sources:
            return {"returncode": 0, "passed": True, "command": "no Python files found",
                    "output": "No Python source files were found; Python compilation was not applicable."}
        command = [sys.executable, "-m", "compileall", "-q", *sources]
        try:
            result = subprocess.run(command, cwd=self.root, capture_output=True,
                                    text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Compile check exceeded {timeout} seconds.") from exc
        return {"returncode": result.returncode, "passed": result.returncode == 0,
                "command": "python -m compileall -q <discovered project Python files>",
                "output": (result.stdout + "\n" + result.stderr)[-8000:]}
