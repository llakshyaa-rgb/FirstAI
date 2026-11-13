"""Small deterministic request router for Orbit's safe local tools.

This router handles only unambiguous local utility intents. All other messages
continue to the language model; no guessed destructive action is executed.
"""
from __future__ import annotations
import re
import hashlib
from .local_tools import WorkspaceTools, safe_calculate
from .tools import ToolRegistry
from .web_search import search_web


class LocalAgent:
    def __init__(self, workspace=None, progress=None, memory=None):
        self.workspace = WorkspaceTools(workspace)
        self.progress = progress or (lambda message, **details: None)
        self.memory = memory
        self.registry = ToolRegistry()
        self.registry.register("calculate", "Evaluate restricted arithmetic", safe_calculate, requires_confirmation=False)
        self.registry.register("web_search", "Search the public web and return source links", search_web, requires_confirmation=False)
        self.registry.register("list_files", "List visible workspace files", self.workspace.list_files, requires_confirmation=False)
        self.registry.register("read_file", "Read a UTF-8 workspace file", self.workspace.read_file, requires_confirmation=False)
        self.registry.register("search_files", "Search visible workspace text", self.workspace.search_files, requires_confirmation=False)
        self.registry.register("create_file", "Create a new workspace file", self.workspace.create_file)
        self.registry.register("update_file", "Update a workspace file after backing it up", self.workspace.update_file)
        self.registry.register("run_tests", "Run the repository unit tests", self.workspace.run_project_tests)
        self.registry.register("compile_project", "Compile-check project modules", self.workspace.compile_project)

    def try_execute(self, message):
        text = message.strip()
        lower = text.casefold()

        if self.memory and lower.startswith("remember that "):
            note = text[len("remember that "):].strip()
            if not note:
                return "Tell me what you'd like me to remember."
            key = "owner_note_" + hashlib.sha256(note.casefold().encode("utf-8")).hexdigest()[:12]
            self.memory.set(key, note)
            return "I'll remember that preference or note on this device."

        if self.memory and lower.startswith(("i prefer ", "i like ", "i love ", "i enjoy ", "i don't like ", "i dislike ")):
            key = "preference_" + hashlib.sha256(lower.encode("utf-8")).hexdigest()[:12]
            self.memory.set(key, text)
            return "Noted. I'll use that preference in future conversations on this device."

        if self.memory and lower in {"what do you remember", "what do you remember about me", "show my preferences"}:
            items = self.memory.get_all()
            if not items:
                return "I don't have any saved preferences or notes yet."
            return "Saved preferences and notes:\n" + "\n".join(
                f"- {item['value']}" for item in items
            )

        if self.memory and lower.startswith("forget:"):
            key = text.split(":", 1)[1].strip()
            if not key:
                return "Give me the exact memory key to remove."
            return "Memory removed." if self.memory.delete(key) else "I couldn't find that memory key."

        match = re.match(r"^(?:calculate|compute|what is|what's)\s+(.+?)\s*[?]?$", text, re.I)
        if match and re.search(r"\d", match.group(1)) and re.fullmatch(r"[\d\s()+\-*/%.^]+", match.group(1)):
            expression = match.group(1).replace("^", "**")
            self.progress("I'll calculate that directly.")
            result = self.registry.call("calculate", {"expression": expression})
            return f"{expression} = {result}"

        match = re.match(r"^(?:search the web for|search web for|look up|research)\s+(.+?)\s*$", text, re.I)
        if match:
            query = match.group(1).strip()
            self.progress("I'll search the web and gather source links.")
            results = self.registry.call("web_search", {"query": query})
            if not results:
                return "The web search returned no results."
            return "Web research results:\n" + "\n\n".join(
                f"- {item['title']} ({item['source']})\n  {item['description']}\n  {item['url']}" for item in results
            )

        if lower in {"list files", "show files", "list project files", "show project files"}:
            self.progress("I'll inspect the project workspace.")
            items = self.registry.call("list_files", {"relative": "."})
            return "Project files:\n" + ("\n".join(
                ("[dir] " if item["type"] == "directory" else "[file] ") + item["name"]
                for item in items
            ) if items else "(No visible files found.)")

        match = re.match(r"^(?:create|make)\s+file:\s*([^\n]+)\n([\s\S]+)$", text, re.I)
        if match:
            relative, content = match.group(1).strip(), match.group(2)
            self.progress(f"I'll create {relative} inside the workspace.")
            result = self.registry.call("create_file", {"relative": relative, "content": content}, confirmed=True)
            return f"Created {result['path']} ({result['bytes']:,} bytes)."

        match = re.match(r"^update\s+file:\s*([^\n]+)\n([\s\S]+)$", text, re.I)
        if match:
            relative, content = match.group(1).strip(), match.group(2)
            self.progress(f"I'll update {relative}; the previous version will be backed up first.")
            result = self.registry.call("update_file", {"relative": relative, "content": content}, confirmed=True)
            backup = result["backup"] or "no backup was needed"
            return f"Updated {result['path']} ({result['bytes']:,} bytes). Backup: {backup}."

        match = re.match(r"^(?:read|show|open)\s+(?:the\s+)?file\s+([^\s]+)\s*$", text, re.I)
        if match:
            relative = match.group(1)
            self.progress(f"I'll read {relative}.")
            content = self.registry.call("read_file", {"relative": relative})
            return content[:16000] + ("\n[Output truncated at 16,000 characters.]" if len(content) > 16000 else "")

        match = re.match(r"^(?:search|find)\s+(?:the\s+)?project\s+for\s+(.+?)\s*$", text, re.I)
        if match:
            query = match.group(1).strip('"\' ')
            self.progress(f"I'll search the workspace for {query!r}.")
            matches = self.registry.call("search_files", {"query": query})
            if not matches:
                return f"I couldn't find {query!r} in the visible project files."
            return "Matches:\n" + "\n".join(
                f"{item['path']}:{item['line']}: {item['text']}" for item in matches
            )

        if lower in {"run tests", "run the tests", "test the project", "run project tests"}:
            self.progress("I'll run the repository's unittest suite.")
            result = self.registry.call("run_tests", confirmed=True)
            status = "passed" if result["passed"] else "failed"
            return f"Project tests {status} (exit {result['returncode']}).\n{result['output']}"

        if lower in {"check syntax", "compile project", "check project syntax"}:
            self.progress("I'll compile-check the project modules.")
            result = self.registry.call("compile_project", confirmed=True)
            status = "passed" if result["passed"] else "failed"
            return f"Project compile check {status} (exit {result['returncode']}).\n{result['output']}"

        return None
