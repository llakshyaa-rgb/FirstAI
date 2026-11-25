"""Ollama-backed Qwen runtime implementing Orbit's existing chat interface.

This keeps Orbit's identity, history, memory and tool-planning flow while
delegating text generation to an Ollama-served GGUF model.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from contextlib import contextmanager

from .identity import build_system_prompt
from .runtime import ModelNotReady


class OllamaOrbitRuntime:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 max_history_messages: int = 8, max_new_tokens: int = 384):
        self.model_name = (model or os.environ.get(
            "ORBIT_OLLAMA_MODEL", "firstai-qwen3-8b"
        )).strip()
        self.base_url = (base_url or os.environ.get(
            "ORBIT_OLLAMA_URL", "http://127.0.0.1:11434"
        )).rstrip("/")
        self.max_history_messages = max(0, int(max_history_messages))
        self.max_new_tokens = max(16, min(
            2048, int(os.environ.get("ORBIT_MAX_NEW_TOKENS", str(max_new_tokens)))
        ))
        self.context_length = max(2048, min(
            32768, int(os.environ.get("ORBIT_CONTEXT_LENGTH", "8192"))
        ))
        self.timeout = max(30, int(os.environ.get("ORBIT_OLLAMA_TIMEOUT", "600")))
        self.keep_alive = os.environ.get("ORBIT_OLLAMA_KEEP_ALIVE", "5m")
        self.load_error = None
        self.instruction_tuned = True
        self.instruction_tuning_steps = 0

    @property
    def ready(self) -> bool:
        try:
            request = urllib.request.Request(self.base_url + "/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read(2_000_000).decode("utf-8"))
            names = {
                str(item.get("name", "")).split(":")[0]
                for item in data.get("models", [])
            }
            wanted = self.model_name.split(":")[0]
            ready = wanted in names or any(name.startswith(wanted) for name in names)
            self.load_error = None if ready else (
                f"Ollama is running, but model '{self.model_name}' is not installed. "
                "Create/pull the model and set ORBIT_OLLAMA_MODEL."
            )
            return ready
        except Exception as exc:
            self.load_error = (
                f"Cannot reach Ollama at {self.base_url}: {type(exc).__name__}: {exc}"
            )
            return False

    def unload(self):
        """Ask Ollama to release Orbit's chat model so a specialist can use the GPU."""
        payload = {"model": self.model_name, "keep_alive": 0, "stream": False}
        request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read(4096)
            return True
        except Exception as exc:
            raise ModelNotReady(
                f"Could not unload chat model '{self.model_name}' before specialist GPU work: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @contextmanager
    def offload_for_external_gpu_work(self):
        # The next normal reply automatically reloads the configured chat model.
        self.unload()
        yield

    def _chat(self, messages, max_tokens=None, temperature=0.7):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            # Qwen3 otherwise spends time producing long reasoning traces.
            # Keep ordinary Orbit chat fast; deeper reasoning can be added
            # selectively for explicitly complex tasks in a later pass.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.context_length,
                "num_predict": max(1, min(4096, int(max_tokens or self.max_new_tokens))),
                "temperature": float(temperature),
                "top_p": 0.9,
                "repeat_penalty": 1.08,
            },
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read(8_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode("utf-8", "replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ModelNotReady(
                f"Could not call Ollama at {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc
        message = result.get("message") or {}
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama returned an empty assistant message.")
        return answer.strip()

    def _messages(self, message, history=None, owner_preferences=None):
        # Use the complete canonical Orbit contract, not the abbreviated
        # personality summary, so the connected model gets Orbit's full style,
        # autonomy, maturity, privacy and media instructions.
        system = build_system_prompt(owner_preferences=owner_preferences)
        messages = [{"role": "system", "content": system}]
        if history:
            for item in history[-self.max_history_messages:]:
                role = "assistant" if item.get("role") in {"assistant", "orbit"} else "user"
                content = str(item.get("content", "")).strip()
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": str(message).strip()})
        return messages

    def reply(self, message, history=None, owner_preferences=None,
              on_progress=None, max_new_tokens=None):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        if on_progress:
            on_progress({"kind": "status", "message": "Orbit is generating with its connected Qwen model."})
        answer = self._chat(
            self._messages(message, history, owner_preferences),
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.7,
        )
        if on_progress:
            on_progress({
                "kind": "partial",
                "message": answer,
                "tokens_generated": 0,
                "strategy": "ollama",
            })
        return answer

    def plan_tool_call(self, message, tool_descriptions):
        safe = [
            item for item in tool_descriptions
            if item.get("name") in {
                "calculate", "web_search", "live_data", "list_files",
                "read_file", "search_files"
            } and not item.get("requires_confirmation", True)
        ]
        instruction = (
            "You are Orbit's constrained tool planner. Decide whether a listed "
            "read-only tool is needed. Return exactly one JSON object on one line "
            'with schema {"steps":[{"tool":"tool_name","arguments":{}}]}. '
            "Use one to five steps. If no tool is needed, return an empty steps array. "
            "Never invent tools or arguments.\nOwner request: "
            + str(message)[:5000] + "\nAvailable tools: "
            + json.dumps(safe, ensure_ascii=False) + "\nJSON:"
        )
        try:
            return self._chat(
                [
                    {"role": "system", "content": "Return valid JSON only. No markdown."},
                    {"role": "user", "content": instruction},
                ],
                max_tokens=384,
                temperature=0.0,
            )
        except Exception:
            return ""
