"""Utilities for parameterising a ComfyUI API workflow JSON template."""
from __future__ import annotations
import copy
import json
import math
import secrets
from pathlib import Path


def load_workflow_template(path, prompt, negative_prompt="", seed=None,
                           checkpoint="model.safetensors", width=768, height=768,
                           steps=24, cfg=7.0):
    """Load an API workflow and replace Orbit placeholders with typed values.

    Text placeholders: {{PROMPT}}, {{NEGATIVE_PROMPT}}, {{CHECKPOINT}}
    Numeric placeholders: {{SEED}}, {{WIDTH}}, {{HEIGHT}}, {{STEPS}}, {{CFG}}
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Image prompt must be non-empty text")
    if not isinstance(negative_prompt, str):
        raise ValueError("Negative prompt must be text")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("Checkpoint must be a model filename")
    if type(seed) is not int and seed is not None:
        raise ValueError("Seed must be an integer")
    if seed is not None and not 0 <= seed <= 2**64 - 1:
        raise ValueError("Seed must be between 0 and 2^64-1")
    if type(width) is not int or type(height) is not int or not (64 <= width <= 2048 and 64 <= height <= 2048) or width % 8 or height % 8:
        raise ValueError("Image dimensions must be integers between 64 and 2048")
    if type(steps) is not int or not 1 <= steps <= 150:
        raise ValueError("Steps must be an integer between 1 and 150")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(float(cfg)) or not 0 <= cfg <= 30:
        raise ValueError("CFG must be a finite number between 0 and 30")

    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow template must be a non-empty ComfyUI API workflow object")
    resolved_seed = seed if seed is not None else secrets.randbits(64)
    replacements = {
        "{{PROMPT}}": prompt,
        "{{NEGATIVE_PROMPT}}": negative_prompt,
        "{{CHECKPOINT}}": checkpoint,
        "{{SEED}}": int(resolved_seed),
        "{{WIDTH}}": width,
        "{{HEIGHT}}": height,
        "{{STEPS}}": steps,
        "{{CFG}}": float(cfg),
    }

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            for marker in ("{{PROMPT}}", "{{NEGATIVE_PROMPT}}", "{{CHECKPOINT}}"):
                value = value.replace(marker, replacements[marker])
        return value

    return replace(copy.deepcopy(workflow))
