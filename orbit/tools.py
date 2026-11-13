"""Explicit allow-list tool registry with authorization gates.

No arbitrary shell, Python eval, or unrestricted file access is exposed.
Applications register narrow, purpose-built callables.
"""
from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    handler: object
    requires_confirmation: bool = True

class ToolRegistry:
    def __init__(self):
        self._tools={}
    def register(self,name,description,handler,requires_confirmation=True):
        if not name or not name.replace("_","").isalnum():
            raise ValueError("Tool name must contain only letters, digits, and underscores")
        if not callable(handler):
            raise TypeError("Tool handler must be callable")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name]=Tool(name,description,handler,bool(requires_confirmation))
    def describe(self):
        return [{"name":t.name,"description":t.description,
                 "requires_confirmation":t.requires_confirmation}
                for t in self._tools.values()]
    def call(self,name,arguments=None,confirmed=False):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        tool=self._tools[name]
        if tool.requires_confirmation and not confirmed:
            raise PermissionError(f"Tool '{name}' requires explicit confirmation")
        return tool.handler(**(arguments or {}))
