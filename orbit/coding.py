"""Bounded autonomous coding workflow: plan, edit with backups, check, repair."""
from __future__ import annotations
import json
import re
import shutil
import subprocess
from pathlib import Path

MAX_FILES = 5
MAX_BYTES = 48000
MAX_REPAIRS = 2
BLOCKED = {".git", ".venv", "venv", "__pycache__", ".env", ".secrets",
           ".orbit_backups", "node_modules", "dist", "build"}


def extract_json(raw):
    text = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", str(raw or "").strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The coding model did not return a JSON object.")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")
    return result


class AutonomousCoder:
    def __init__(self, runtime, workspace, progress=None):
        self.runtime = runtime
        self.workspace = workspace
        self.progress = progress or (lambda message, **details: None)

    def _reply(self, prompt, tokens=1200):
        return self.runtime.reply(prompt, history=[], owner_preferences="",
            max_new_tokens=tokens,
            on_progress=lambda event: self.progress(event.get("message", "Working…"),
                kind=event.get("kind", "status"),
                **{k: v for k, v in event.items() if k not in ("kind", "message")}))

    def _path(self, relative):
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Missing relative file path.")
        path = Path(relative)
        if path.is_absolute() or any(p in BLOCKED or p.startswith(".env") for p in path.parts):
            raise PermissionError("Protected path selected.")
        target = (self.workspace.root / path).resolve()
        if target != self.workspace.root and self.workspace.root not in target.parents:
            raise PermissionError("Path escapes the selected workspace.")
        if target.suffix.lower() not in {".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
            ".html", ".css", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".sql"}:
            raise PermissionError("File type is not in the coding allow-list.")
        return target

    def _read(self, paths):
        result = []
        for relative in paths[:MAX_FILES]:
            target = self._path(relative)
            if not target.is_file():
                result.append({"path": relative, "exists": False, "content": ""})
            elif target.stat().st_size <= MAX_BYTES:
                try:
                    content = target.read_text(encoding="utf-8")
                    secret_pattern = re.compile(
                        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|private[_-]?key)\\s*[:=]\\s*[\"'][^\"']{8,}[\"']"
                    )
                    if secret_pattern.search(content):
                        self.progress("Skipping " + relative + " because it appears to contain credentials.")
                        continue
                    result.append({"path": relative, "exists": True, "content": content})
                except UnicodeDecodeError:
                    self.progress("Skipping non-UTF-8 file " + relative)
            else:
                self.progress("Skipping oversized file " + relative)
        return result

    def _paths_from_plan(self, plan):
        files = plan.get("files")
        if not isinstance(files, list):
            raise ValueError("The coding plan did not list target files.")
        paths, seen = [], set()
        for item in files[:MAX_FILES]:
            relative = item if isinstance(item, str) else item.get("path") if isinstance(item, dict) else None
            target = self._path(relative)
            normalized = target.relative_to(self.workspace.root).as_posix()
            if normalized not in seen:
                paths.append(normalized)
                seen.add(normalized)
        if not paths:
            raise ValueError("No safe target files were selected.")
        return paths

    def _apply(self, raw, paths):
        payload = extract_json(raw)
        edits = payload.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("No file edits were returned.")
        changed = []
        for edit in edits[:MAX_FILES]:
            if not isinstance(edit, dict) or edit.get("path") not in paths:
                raise PermissionError("The model attempted an unplanned file edit.")
            relative = edit["path"]
            target = self._path(relative)
            if not target.exists():
                content = edit.get("content")
                if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > MAX_BYTES:
                    raise ValueError("New files require complete, bounded text content.")
                changed.append(self.workspace.create_file(relative, content))
                continue
            if "patches" in edit:
                patches = edit["patches"]
                if not isinstance(patches, list) or not patches:
                    raise ValueError("A file patch must contain exact replacements.")
                content = target.read_text(encoding="utf-8")
                for patch in patches[:12]:
                    if not isinstance(patch, dict) or not isinstance(patch.get("old"), str) or not patch["old"]:
                        raise ValueError("A patch needs a non-empty exact old-text anchor.")
                    new = patch.get("new")
                    if not isinstance(new, str):
                        raise ValueError("Patch replacement text must be a string.")
                    occurrences = content.count(patch["old"])
                    if occurrences != 1:
                        raise ValueError("Patch anchor for " + relative + " matched " + str(occurrences) + " times; refusing an ambiguous edit.")
                    content = content.replace(patch["old"], new, 1)
                if len(content.encode("utf-8")) > MAX_BYTES:
                    raise ValueError("Patched file exceeds the safe file-size limit.")
                changed.append(self.workspace.update_file(relative, content))
                continue
            raise ValueError("Existing files must be changed with exact, targeted patches rather than full-file rewrites.")
        if not changed:
            raise ValueError("No file changes were applied.")
        return changed, str(payload.get("summary") or "Applied targeted changes.")

    def _checks(self, changed):
        compile_result = self.workspace.compile_project(timeout=120)
        javascript = []
        node = shutil.which("node")
        if node:
            for item in changed:
                relative = item.get("path", "")
                if Path(relative).suffix.lower() in {".js", ".mjs", ".cjs"}:
                    result = subprocess.run([node, "--check", relative], cwd=self.workspace.root,
                                            capture_output=True, text=True, timeout=60, check=False)
                    javascript.append({"path": relative, "passed": result.returncode == 0,
                                       "output": (result.stdout + "\\n" + result.stderr)[-3000:]})
        tests = None
        if compile_result.get("passed") and all(item["passed"] for item in javascript) and (self.workspace.root / "tests").is_dir():
            tests = self.workspace.run_project_tests(timeout=180)
        ok = bool(compile_result.get("passed")) and all(item["passed"] for item in javascript) and (
            tests is None or bool(tests.get("passed"))
        )
        return {"ok": ok, "compile": compile_result, "javascript": javascript, "tests": tests}

    @staticmethod
    def _failures(checks):
        return "\n\n".join(
            key.upper() + " FAILURE:\n" + str(value.get("output", ""))[-6500:]
            for key in ("compile", "tests")
            if (value := checks.get(key)) and not value.get("passed")
        ) or "No check failures."

    def run(self, goal):
        if not self.runtime.ready:
            raise RuntimeError("A trained FirstGPT checkpoint is required for autonomous coding.")
        self.progress("Coding agent: mapping the project and choosing targeted files.")
        overview = self.workspace.inspect_project(".", max_files=80)
        plan = extract_json(self._reply(
            "Plan a minimal implementation for this goal using only paths in this project map. "
            "Return ONLY JSON: " + '{"files":["relative/path"],"plan":"ordered short plan"}. '
            "Choose at most five files. Exclude secrets, dependencies, generated files and lockfiles.\n"
            "GOAL: " + goal[:2000] + "\nPROJECT MAP: " + json.dumps(overview, ensure_ascii=False)[:9000], 500))
        paths = self._paths_from_plan(plan)
        context = self._read(paths)
        if not context:
            raise RuntimeError("No safe readable source files were available.")
        self.progress("Plan: " + str(plan.get("plan", "Implementing targeted changes."))[:600], files=paths)
        prompt = (
            "Implement the owner's goal in these files. Preserve unrelated behaviour and interfaces. "
            "Return ONLY valid JSON: " + '{"summary":"...","edits":[{"path":"existing/path","patches":[{"old":"exact unique text","new":"replacement text"}]},{"path":"new/path","content":"complete new file text"}]}. '
            "For existing files use exact, unique, small patches; never rewrite entire files. Each old anchor must occur exactly once. "
            "New files may use complete content. Never edit paths outside the supplied list.\nGOAL: " + goal[:2000] +
            "\nPLAN: " + str(plan.get("plan", ""))[:800] +
            "\nALLOWED PATHS: " + json.dumps(paths) +
            "\nCURRENT FILES: " + json.dumps(context, ensure_ascii=False)[:22000])
        raw = self._reply(prompt, 1500)
        for edit_attempt in range(2):
            try:
                changed, summary = self._apply(raw, paths)
                break
            except Exception as exc:
                if edit_attempt:
                    raise
                self.progress("The proposed patch was rejected (" + type(exc).__name__ + "). Asking the coding agent to correct its patch before touching the files.",
                              kind="method_failed")
                raw = self._reply(
                    prompt + "\\nPREVIOUS PATCH REJECTED: " + type(exc).__name__ + ": " + str(exc)[:1200] +
                    "\\nReturn a corrected JSON patch. Use exact unique anchors copied from the current files.",
                    1500
                )
        self.progress("Applied changes with automatic backups.", changes=changed)
        checks = None
        for attempt in range(MAX_REPAIRS + 1):
            self.progress("Running compile and available project checks, pass " + str(attempt + 1) + ".")
            checks = self._checks(changed)
            if checks["ok"]:
                self.progress("Implementation checks passed.")
                return {"status": "completed", "summary": summary, "plan": plan.get("plan", ""),
                        "changes": changed, "checks": checks, "repair_rounds": attempt}
            if attempt == MAX_REPAIRS:
                break
            self.progress("A check failed. Diagnosing it and preparing a repair.", kind="method_failed")
            current = self._read(paths)
            repair = (
                "Repair the implementation using the actual failure output. Keep the requested behaviour. "
                "Return ONLY valid JSON: " + '{"summary":"...","edits":[{"path":"existing/path","patches":[{"old":"exact unique text","new":"replacement text"}]},{"path":"new/path","content":"complete new file text"}]}. '
                "Use exact, unique small patches for existing files; never rewrite them wholesale. Only these paths may change: " + json.dumps(paths) +
                "\nGOAL: " + goal[:1600] + "\nFAILURES: " + self._failures(checks)[-8500:] +
                "\nCURRENT FILES: " + json.dumps(current, ensure_ascii=False)[:20000])
            repair_raw = self._reply(repair, 1500)
            for repair_attempt in range(2):
                try:
                    fixed, summary = self._apply(repair_raw, paths)
                    break
                except Exception as exc:
                    if repair_attempt:
                        raise
                    self.progress("The repair patch was rejected; requesting a corrected repair.", kind="method_failed")
                    repair_raw = self._reply(
                        repair + "\\nPREVIOUS REPAIR REJECTED: " + type(exc).__name__ + ": " + str(exc)[:1200] +
                        "\\nReturn a corrected exact patch.",
                        1500
                    )
            changed.extend(fixed)
            self.progress("Applied a repair with fresh backups.", changes=fixed)
        return {"status": "needs_attention", "summary": summary, "plan": plan.get("plan", ""),
                "changes": changed, "checks": checks, "repair_rounds": MAX_REPAIRS}
