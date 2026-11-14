"""Bounded autonomous coding workflow: plan, edit with backups, check, repair."""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

MAX_FILES = 8
MAX_BYTES = 180000
MAX_REPAIRS = 3
ENGINEERING_ROLES = {
    "architect": "Own architecture, dependencies, interfaces and implementation order. Prefer small compatible changes.",
    "designer": "Own product-quality UX and visual design: establish a coherent design system, strong hierarchy, deliberate typography, spacing, color and component states; make every screen responsive and accessible; avoid generic template layouts; design complete user journeys, empty/loading/error/success states, forms, navigation and realistic content. For adult-oriented products, design for adults with privacy, consent, clear boundaries and age-appropriate access controls.",
    "frontend": "Own browser UI, state, interactions, validation, accessibility and responsive layouts.",
    "backend": "Own API routes, data models, persistence, error handling and service boundaries.",
    "data": "Own data ingestion, schemas, provenance, validation, missing values and export correctness.",
    "security": "Own trust boundaries, input validation, path safety, secret handling and least privilege.",
    "qa": "Own acceptance criteria, regression risks and observable correctness; do not claim checks were run unless they were.",
    "implementation": "Implement the assigned scoped change while preserving existing interfaces and behaviour.",
}
BLOCKED = {".git", ".venv", "venv", "__pycache__", ".env", ".secrets",
           ".orbit_backups", ".orbit_agent_workspaces", "node_modules", "dist", "build"}


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
    def __init__(self, runtime, workspace, progress=None, approval_required=None):
        self.runtime = runtime
        self.workspace = workspace
        self.progress = progress or (lambda message, **details: None)
        self.approval_required = approval_required
        if not callable(self.approval_required):
            raise ValueError("Autonomous coding requires an explicit owner-approval callback.")

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
        if path.is_absolute() or ".." in path.parts or any(p in BLOCKED or p.startswith(".env") for p in path.parts):
            raise PermissionError("Protected path selected.")
        current = self.workspace.root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise PermissionError("Symbolic links are excluded from autonomous coding.")
        target = (self.workspace.root / path).resolve()
        if target != self.workspace.root and self.workspace.root not in target.parents:
            raise PermissionError("Path escapes the selected workspace.")
        sensitive_names = {"credentials", "secrets", "secret", "tokens", "token",
                           "id_rsa", "id_ed25519", "private_key", "service_account"}
        if any(part.casefold() in sensitive_names for part in path.parts):
            raise PermissionError("Credential-related paths are excluded from autonomous coding.")
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
                        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|private[_-]?key)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
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
        prepared, seen = [], set()
        for edit in edits[:MAX_FILES]:
            if not isinstance(edit, dict) or edit.get("path") not in paths:
                raise PermissionError("The model attempted an unplanned file edit.")
            relative = edit["path"]
            if relative in seen:
                raise ValueError("A file may appear only once in an edit batch.")
            seen.add(relative)
            target = self._path(relative)
            if not target.exists():
                content = edit.get("content")
                if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > MAX_BYTES:
                    raise ValueError("New files require complete, bounded text content.")
                prepared.append((relative, content, False))
                continue
            if "patches" not in edit:
                raise ValueError("Existing files must use exact, targeted patches rather than full-file rewrites.")
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
            prepared.append((relative, content, True))
        if not prepared:
            raise ValueError("No valid file edits were supplied.")
        changed = []
        for relative, content, existed in prepared:
            if existed:
                changed.append(self.workspace.update_file(relative, content))
            else:
                changed.append(self.workspace.create_file(relative, content))
        return changed, str(payload.get("summary") or "Applied targeted changes.")

    def _project_root_for_changes(self, changed):
        candidates = []
        for item in changed if isinstance(changed, list) else []:
            relative = item.get("path", "") if isinstance(item, dict) else ""
            try:
                target = self._path(relative)
            except (ValueError, PermissionError):
                continue
            parent = target.parent
            while parent == self.workspace.root or self.workspace.root in parent.parents:
                package = parent / "package.json"
                if package.is_file():
                    try:
                        data = json.loads(package.read_text(encoding="utf-8"))
                        scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
                        if isinstance(scripts, dict) and any(
                            isinstance(scripts.get(key), str) for key in ("build", "test", "dev", "start")
                        ):
                            candidates.append(parent)
                    except (OSError, ValueError):
                        pass
                if parent == self.workspace.root:
                    break
                parent = parent.parent
        if not candidates:
            return None
        return max(candidates, key=lambda path: len(path.parts))

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
                                       "output": (result.stdout + "\n" + result.stderr)[-3000:]})
        project_root = self._project_root_for_changes(changed)
        build = None
        build_enabled = os.environ.get("ORBIT_AUTORUN_BUILD", "").strip().lower() in {"1", "true", "yes"}
        if build_enabled and project_root is None:
            build = {"passed": True, "skipped": True, "reason": "No changed runnable package project was detected."}
        elif build_enabled and project_root is not None and not (compile_result.get("passed") and all(item["passed"] for item in javascript)):
            build = {"passed": True, "skipped": True, "reason": "Skipped because an earlier compile or JavaScript check failed."}
        elif build_enabled and project_root is not None:
            package = project_root / "package.json"
            try:
                package_data = json.loads(package.read_text(encoding="utf-8"))
                scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
                if isinstance(scripts, dict) and isinstance(scripts.get("build"), str):
                    manager = ("pnpm" if (project_root / "pnpm-lock.yaml").exists() else
                               "yarn" if (project_root / "yarn.lock").exists() else
                               "bun" if (project_root / "bun.lockb").exists() or (project_root / "bun.lock").exists() else "npm")
                    executable = shutil.which(manager)
                    if not executable:
                        build = {"passed": False, "output": manager + " is not installed.", "project_root": str(project_root)}
                    else:
                        result = subprocess.run([executable, "run", "build"], cwd=project_root,
                            capture_output=True, text=True, timeout=240, check=False, shell=False)
                        build = {"passed": result.returncode == 0,
                                 "output": (result.stdout + "\n" + result.stderr)[-6000:],
                                 "project_root": str(project_root), "command": manager + " run build"}
                else:
                    build = {"passed": True, "skipped": True, "reason": "No package.json build script exists.",
                             "project_root": str(project_root)}
            except subprocess.TimeoutExpired:
                build = {"passed": False, "output": "Build exceeded the 240-second time limit.",
                         "project_root": str(project_root)}
            except (OSError, ValueError) as exc:
                build = {"passed": False, "output": type(exc).__name__ + ": " + str(exc),
                         "project_root": str(project_root)}
        tests = None
        tests_enabled = os.environ.get("ORBIT_AUTORUN_TESTS", "").strip().lower() in {"1", "true", "yes"}
        if tests_enabled and compile_result.get("passed") and all(item["passed"] for item in javascript) and (self.workspace.root / "tests").is_dir():
            tests = self.workspace.run_project_tests(timeout=180)
        ok = bool(compile_result.get("passed")) and all(item["passed"] for item in javascript) and (
            tests is None or bool(tests.get("passed"))
        ) and (build is None or bool(build.get("passed")))
        return {"ok": ok, "compile": compile_result, "javascript": javascript, "build": build,
                "build_enabled": build_enabled, "project_root": str(project_root) if project_root else None,
                "tests": tests, "tests_enabled": tests_enabled}

    @staticmethod
    def _failures(checks):
        failures = [
            key.upper() + " FAILURE:\n" + str(value.get("output", ""))[-6500:]
            for key in ("compile", "build", "tests")
            if (value := checks.get(key)) and not value.get("passed")
        ]
        failures.extend(
            "JAVASCRIPT FAILURE (" + str(item.get("path", "file")) + "):\n" +
            str(item.get("output", ""))[-3000:]
            for item in checks.get("javascript", []) if not item.get("passed")
        )
        return "\n\n".join(failures) or "No check failures."

    def run(self, goal):
        if not self.runtime.ready:
            raise RuntimeError("A trained FirstGPT checkpoint is required for autonomous coding.")
        self.progress("Coding agent: mapping the project and planning the implementation in phases.")
        overview = self.workspace.inspect_project(".", max_files=80)
        map_summary = {
            "files": [item.get("path") for item in overview.get("files", [])[:80]],
            "symbols": [
                {"path": item.get("path"),
                 "names": [symbol.get("name") for symbol in item.get("definitions", [])[:12]]}
                for item in overview.get("python_source_map", []) if item.get("definitions")
            ][:35],
        }
        plan = extract_json(self._reply(
            "You are the lead engineer. Decompose the owner's goal into a short dependency-ordered plan. "
            "Return ONLY JSON: " + '{"phases":[{"role":"architect|designer|frontend|backend|data|security|qa|implementation","goal":"one concrete phase outcome","files":["relative/path"]}]}. '
            "Use up to eight phases and at most eight files per phase. Assign the most relevant specialist role to each phase; "
            "separate architecture, UX, frontend, backend, data, security and QA concerns when genuinely distinct. Plan integration contracts and dependency order. Use only existing paths from the project map "
            "or a clearly named new source file. Keep phases small and independently checkable. Exclude secrets, "
            "dependencies, generated files and lockfiles.\nPROJECT MAP: " +
            json.dumps(map_summary, ensure_ascii=False)[:6500] +
            "\nOWNER GOAL: " + goal[:2200], 800))
        phases = plan.get("phases")
        if not isinstance(phases, list) or not phases:
            if isinstance(plan.get("files"), list):
                phases = [{"goal": plan.get("plan", goal), "files": plan["files"]}]
            else:
                raise ValueError("The engineering plan did not contain executable phases.")
        phases = phases[:8]
        team_root = self.workspace.root / ".orbit_agent_workspaces" / uuid.uuid4().hex
        team_root.mkdir(parents=True, exist_ok=True)
        (team_root / "task.json").write_text(json.dumps({
            "goal": goal, "phase_count": len(phases),
            "note": "Role-scoped task artifacts. Only the lead coding workflow applies patches to the project."
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        all_changes, completed_phases, total_repairs = [], [], 0
        for phase_index, phase in enumerate(phases, 1):
            if not isinstance(phase, dict):
                raise ValueError("An implementation phase must be an object.")
            phase_goal = str(phase.get("goal") or goal)[:1200]
            phase_role = str(phase.get("role") or "implementation").strip().lower()
            if phase_role not in ENGINEERING_ROLES:
                phase_role = "implementation"
            role_guidance = ENGINEERING_ROLES[phase_role]
            paths = self._paths_from_plan({"files": phase.get("files")})
            context = self._read(paths)
            if not context:
                raise RuntimeError("No safe readable files were available for phase " + str(phase_index) + ".")
            # Each role gets its own durable brief and source snapshot. These are
            # isolated task artifacts; the lead remains the sole project writer.
            role_space = team_root / ("phase_" + str(phase_index) + "_" + phase_role)
            role_space.mkdir(parents=True, exist_ok=True)
            (role_space / "brief.json").write_text(json.dumps({
                "role": phase_role, "guidance": role_guidance,
                "phase_goal": phase_goal, "owner_goal": goal,
                "allowed_paths": paths,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            (role_space / "source_snapshot.json").write_text(
                json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
            role_project = role_space / "workspace"
            for source_file in context:
                if source_file.get("exists"):
                    snapshot_file = role_project / source_file["path"]
                    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_file.write_text(source_file.get("content", ""), encoding="utf-8")
            self.progress("Phase " + str(phase_index) + " of " + str(len(phases)) + " · " +
                          phase_role.title() + " specialist: " + phase_goal, files=paths, role=phase_role)
            prompt = (
                "You are Orbit's " + phase_role + " specialist. " + role_guidance + " " +
                "Implement this phase of the owner's project goal. Preserve unrelated behaviour and interfaces. "
                "Return ONLY valid JSON: " + '{"summary":"...","edits":[{"path":"existing/path","patches":[{"old":"exact unique text","new":"replacement text"}]},{"path":"new/path","content":"complete new file text"}]}. '
                "For existing files use exact, unique, small patches; never rewrite entire files. Each old anchor must occur exactly once. "
                "New files may use complete content. Never edit paths outside the supplied list.\nCURRENT FILES: " +
                json.dumps(context, ensure_ascii=False)[:15000] +
                "\nALLOWED PATHS: " + json.dumps(paths) +
                "\nPHASE GOAL: " + phase_goal + "\nOWNER GOAL: " + goal[:1800])
            raw = self._reply(prompt, 1500)
            (role_space / "proposal.json").write_text(json.dumps({
                "raw_proposal": raw, "allowed_paths": paths
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.progress("Reviewer pass: checking the proposed changes for scope, compatibility and regressions.")
            try:
                review = extract_json(self._reply(
                    "You are the independent reviewer for this phase. Review the proposed exact patches against "
                    "the owner's goal and current files. Find concrete correctness, compatibility, security or "
                    "scope problems. Return ONLY JSON: " +
                    '{"approved":true,"issues":["short concrete issue"]}. Set approved false only for a material issue.\n' +
                    "CURRENT FILES: " + json.dumps(context, ensure_ascii=False)[:8000] +
                    "\nPATCHES: " + raw[:5000] +
                    "\nPHASE: " + phase_goal + "\nOWNER GOAL: " + goal[:1200], 450))
                reviewer_space = role_space / "reviewer"
                reviewer_space.mkdir(parents=True, exist_ok=True)
                (reviewer_space / "source_snapshot.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
                (reviewer_space / "proposal.json").write_text(
                    json.dumps({"proposal": raw, "phase_goal": phase_goal},
                               ensure_ascii=False, indent=2), encoding="utf-8")
                (reviewer_space / "review.json").write_text(
                    json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
                (role_space / "review.json").write_text(
                    json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
                issues = review.get("issues", [])
                if not review.get("approved", False) and issues:
                    self.progress("Reviewer found issues; requesting a corrected implementation before writing.",
                                  kind="method_failed", issues=issues[:6])
                    raw = self._reply(prompt + "\nREVIEW FINDINGS: " +
                                      json.dumps(issues[:8], ensure_ascii=False) +
                                      "\nReturn a corrected exact patch batch.", 1500)
                else:
                    self.progress("Reviewer found no blocking issue.")
            except Exception as exc:
                self.progress("Reviewer could not complete (" + type(exc).__name__ +
                              "); strict patch validation and compiler checks remain active.",
                              kind="method_failed")
            # Never write project files until the owner explicitly approves this exact phase.
            proposal_preview = raw[:16000]
            if not self.approval_required(phase_goal, paths, proposal_preview):
                raise PermissionError("Owner denied or did not provide approval. No changes were applied for this phase.")
            for edit_attempt in range(2):
                try:
                    phase_changes, summary = self._apply(raw, paths)
                    break
                except Exception as exc:
                    if edit_attempt:
                        raise
                    self.progress("Patch rejected (" + type(exc).__name__ +
                                  "); asking the implementation agent to correct it.", kind="method_failed")
                    raw = self._reply(prompt + "\nPREVIOUS PATCH REJECTED: " +
                                      type(exc).__name__ + ": " + str(exc)[:1200] +
                                      "\nReturn a corrected exact patch batch.", 1500)
            (role_space / "applied_changes.json").write_text(
                json.dumps({"summary": summary, "changes": phase_changes},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            all_changes.extend(phase_changes)
            self.progress("Phase changes applied with automatic backups.", changes=phase_changes)
            phase_repairs = 0
            checks = None
            for repair_round in range(MAX_REPAIRS + 1):
                self.progress("Validating phase " + str(phase_index) + ", pass " +
                              str(repair_round + 1) + ".")
                checks = self._checks(all_changes)
                if checks["ok"]:
                    phase_repairs = repair_round
                    total_repairs += repair_round
                    completed_phases.append({"phase": phase_index, "goal": phase_goal,
                                             "files": paths, "summary": summary})
                    self.progress("Phase " + str(phase_index) + " passed its available checks.")
                    break
                if repair_round == MAX_REPAIRS:
                    total_repairs += repair_round
                    return {
                        "status": "needs_attention", "summary": summary,
                        "plan": plan, "completed_phases": completed_phases,
                        "failed_phase": phase_index, "changes": all_changes,
                        "checks": checks, "repair_rounds": total_repairs,
                        "agent_workspace": str(team_root.relative_to(self.workspace.root)),
                    }
                self.progress("A check failed. Diagnosing the output and preparing a repair.",
                              kind="method_failed", phase=phase_index, round=repair_round + 1)
                current = self._read(paths)
                repair_prompt = (
                    "Repair this implementation phase using the actual compiler or test failure. "
                    "Preserve the owner's requested behaviour. Return ONLY valid JSON: " +
                    '{"summary":"...","edits":[{"path":"existing/path","patches":[{"old":"exact unique text","new":"replacement text"}]},{"path":"new/path","content":"complete new file text"}]}. '
                    "Use exact, unique small patches for existing files; never rewrite them wholesale. "
                    "Only these paths may change: " + json.dumps(paths) +
                    "\nCURRENT FILES: " + json.dumps(current, ensure_ascii=False)[:10000] +
                    "\nFAILURES: " + self._failures(checks)[-6000:] +
                    "\nPHASE: " + phase_goal + "\nOWNER GOAL: " + goal[:1400])
                repair_raw = self._reply(repair_prompt, 1500)
                repair_preview = repair_raw[:16000]
                if not self.approval_required(
                    "Repair phase " + str(phase_index) + ": " + phase_goal, paths, repair_preview
                ):
                    raise PermissionError(
                        "Owner denied or did not provide approval for the repair. No repair changes were applied."
                    )
                for repair_attempt in range(2):
                    try:
                        fixed, summary = self._apply(repair_raw, paths)
                        break
                    except Exception as exc:
                        if repair_attempt:
                            raise
                        self.progress("Repair patch rejected; requesting a corrected repair.",
                                      kind="method_failed")
                        repair_raw = self._reply(
                            repair_prompt + "\nPREVIOUS REPAIR REJECTED: " +
                            type(exc).__name__ + ": " + str(exc)[:1200] +
                            "\nReturn a corrected exact patch.", 1500)
                        if not self.approval_required(
                            "Corrected repair phase " + str(phase_index) + ": " + phase_goal,
                            paths, repair_raw[:16000]
                        ):
                            raise PermissionError(
                                "Owner did not approve the corrected repair patch. No repair changes were applied."
                            )
                all_changes.extend(fixed)
                self.progress("Repair applied with fresh backups.", changes=fixed)
        return {
            "status": "completed", "summary": "Completed all planned implementation phases.",
            "plan": plan, "completed_phases": completed_phases,
            "changes": all_changes, "checks": checks,
            "repair_rounds": total_repairs,
            "agent_workspace": str(team_root.relative_to(self.workspace.root)),
        }
