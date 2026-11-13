"""Small deterministic request router for Orbit's safe local tools.

This router handles only unambiguous local utility intents. All other messages
continue to the language model; no guessed destructive action is executed.
"""
from __future__ import annotations
import re
from .local_tools import WorkspaceTools, safe_calculate


class LocalAgent:
    def __init__(self, workspace=None, progress=None):
        self.workspace = WorkspaceTools(workspace)
        self.progress = progress or (lambda message, **details: None)

    def try_execute(self, message):
        text = message.strip()
        lower = text.casefold()

        match = re.match(r"^(?:calculate|compute|what is|what's)\s+(.+?)\s*[?]?$", text, re.I)
        if match and re.search(r"\d", match.group(1)) and re.fullmatch(r"[\d\s()+\-*/%.^]+", match.group(1)):
            expression = match.group(1).replace("^", "**")
            self.progress("I'll calculate that directly.")
            result = safe_calculate(expression)
            return f"{expression} = {result}"

        if lower in {"list files", "show files", "list project files", "show project files"}:
            self.progress("I'll inspect the project workspace.")
            items = self.workspace.list_files(".")
            return "Project files:\n" + ("\n".join(
                ("[dir] " if item["type"] == "directory" else "[file] ") + item["name"]
                for item in items
            ) if items else "(No visible files found.)")

        match = re.match(r"^(?:read|show|open)\s+(?:the\s+)?file\s+([^\s]+)\s*$", text, re.I)
        if match:
            relative = match.group(1)
            self.progress(f"I'll read {relative}.")
            content = self.workspace.read_file(relative)
            return content[:16000] + ("\n[Output truncated at 16,000 characters.]" if len(content) > 16000 else "")

        match = re.match(r"^(?:search|find)\s+(?:the\s+)?project\s+for\s+(.+?)\s*$", text, re.I)
        if match:
            query = match.group(1).strip('"\' ')
            self.progress(f"I'll search the workspace for {query!r}.")
            matches = self.workspace.search_files(query)
            if not matches:
                return f"I couldn't find {query!r} in the visible project files."
            return "Matches:\n" + "\n".join(
                f"{item['path']}:{item['line']}: {item['text']}" for item in matches
            )

        if lower in {"run tests", "run the tests", "test the project", "run project tests"}:
            self.progress("I'll run the repository's unittest suite.")
            result = self.workspace.run_project_tests()
            status = "passed" if result["passed"] else "failed"
            return f"Project tests {status} (exit {result['returncode']}).\n{result['output']}"

        if lower in {"check syntax", "compile project", "check project syntax"}:
            self.progress("I'll compile-check the project modules.")
            result = self.workspace.compile_project()
            status = "passed" if result["passed"] else "failed"
            return f"Project compile check {status} (exit {result['returncode']}).\n{result['output']}"

        return None
