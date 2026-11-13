"""Connect the trained FirstGPT checkpoint to Orbit's identity and chat memory.

This runtime is deliberately honest about the baseline model: identity conditioning
does not make an untrained checkpoint instruction-capable.
"""
from __future__ import annotations
import json
import math
import threading
from pathlib import Path
import torch
from torch.nn import functional as F
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer
from .identity import build_system_prompt

class ModelNotReady(RuntimeError):
    pass

class OrbitRuntime:
    def __init__(self, checkpoint="checkpoints/firstgpt.best.pt", device=None,
                 max_history_messages=8, max_new_tokens=120):
        self.checkpoint_path = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_messages = max_history_messages
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.load_error = None
        self.load()

    def load(self):
        if not self.checkpoint_path.is_file():
            self.load_error = f"Checkpoint not found: {self.checkpoint_path}"
            return False
        try:
            saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
            config = GPTConfig(**saved["config"])
            model = FirstGPT(config).to(self.device)
            model.load_state_dict(saved["model"])
            model.eval()
            self.model, self.tokenizer = model, tokenizer
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def ready(self):
        return self.model is not None and self.tokenizer is not None

    def reply(self, message, history=None, on_progress=None):
        if not self.ready:
            raise ModelNotReady(self.load_error or "No trained checkpoint is loaded.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        history = (history or [])[-self.max_history_messages:]
        transcript = build_system_prompt() + "\n\n"
        for item in history:
            role = "Owner" if item.get("role") == "user" else "Orbit"
            transcript += f"{role}: {item.get('content', '')}\n"
        transcript += f"Owner: {message.strip()}\nOrbit:"
        ids = self.tokenizer.encode(transcript)
        # Preserve the newest request and a compact identity prefix within context.
        limit = self.model.config.block_size
        if len(ids) > limit:
            identity_ids = self.tokenizer.encode("You are Orbit. Be composed, clear, honest, and helpful.\n")
            newest_ids = self.tokenizer.encode(f"Owner: {message.strip()}\nOrbit:")
            budget = max(1, limit - len(identity_ids) - len(newest_ids))
            ids = identity_ids[:max(0, limit-len(newest_ids)-1)] + ids[-(budget+len(newest_ids)):]
            ids = ids[-limit:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = []
        with self._lock, torch.inference_mode():
            for index in range(self.max_new_tokens):
                context = input_ids[:, -self.model.config.block_size:]
                logits, _ = self.model(context)
                next_logits = logits[:, -1, :].float()
                # Greedy decoding is stable for a tiny baseline and avoids invalid sampling.
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                token = int(next_id.item())
                generated.append(token)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                if token in (10, 13):
                    break
                if on_progress and (index + 1) % 12 == 0:
                    on_progress({"kind": "generation_progress", "message": "I'm composing the response.", "tokens_generated": index + 1})
        answer = self.tokenizer.decode(generated).strip()
        if not answer:
            answer = "I have a trained checkpoint loaded, but it did not produce a usable response. The current model needs further training."
        return answer
