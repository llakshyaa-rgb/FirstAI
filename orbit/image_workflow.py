"""Utilities for parameterising a ComfyUI API workflow JSON template."""
from __future__ import annotations
import copy
import json
from pathlib import Path


def load_workflow_template(path, prompt, negative_prompt="", seed=None):
    """Load a ComfyUI API-format workflow and replace documented placeholders.

    String placeholders:
      {{PROMPT}}          positive prompt
      {{NEGATIVE_PROMPT}} negative prompt
      {{SEED}}            integer sampler seed
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Image prompt must be non-empty text")
    if not isinstance(negative_prompt, str):
        raise ValueError("Negative prompt must be text")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**64 - 1):
        raise ValueError("Seed must be an integer between 0 and 2^64-1")
    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow template must be a non-empty ComfyUI API workflow object")
    replacements = {
        "{{PROMPT}}": str(prompt),
        "{{NEGATIVE_PROMPT}}": str(negative_prompt),
        "{{SEED}}": str(seed if seed is not None else 0),
    }

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            if value == "{{SEED}}":
                return int(seed if seed is not None else 0)
            for marker, replacement in replacements.items():
                if marker != "{{SEED}}":
                    value = value.replace(marker, replacement)
        return value

    return replace(copy.deepcopy(workflow))
