"""Connect a trained FirstGPT checkpoint to Orbit's identity and chat memory.

Identity conditioning does not make an untrained checkpoint instruction-capable.
Generation retries use a different decoding strategy; they cannot repair missing
knowledge or replace a better-trained model.
"""
from __future__ import annotations
import threading
from pathlib import Path
import torch
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

    def _generate_once(self, prompt_ids, mode, on_progress=None):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = []
        with torch.inference_mode():
            for index in range(self.max_new_tokens):
                context = input_ids[:, -self.model.config.block_size:]
                logits, _ = self.model(context)
                next_logits = logits[:, -1, :].float()
                if mode == "sample":
                    next_logits = next_logits / 0.85
                    top_k = min(40, next_logits.size(-1))
                    values, _ = torch.topk(next_logits, top_k)
                    next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], -float("inf"))
                    probabilities = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                token = int(next_id.item())
                generated.append(token)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                if token in (10, 13):
                    break
                if on_progress and (index + 1) % 12 == 0:
                    on_progress({
                        "kind": "generation_progress",
                        "message": "I'm composing the response.",
                        "tokens_generated": index + 1,
                        "strategy": mode,
                    })
        return self.tokenizer.decode(generated).strip()

    def reply(self, message, history=None, owner_preferences=None, on_progress=None):
        if not self.ready:
            raise ModelNotReady(self.load_error or "No trained checkpoint is loaded.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        history = (history or [])[-self.max_history_messages:]
        transcript = build_system_prompt(owner_preferences=owner_preferences) + "\n\n"
        for item in history:
            role = "Owner" if item.get("role") == "user" else "Orbit"
            transcript += f"{role}: {item.get('content', '')}\n"
        transcript += f"Owner: {message.strip()}\nOrbit:"
        ids = self.tokenizer.encode(transcript)
        limit = self.model.config.block_size
        if len(ids) > limit:
            identity_ids = self.tokenizer.encode(
                "You are Orbit: composed, clear, candid, and helpful. "
                "Answer directly. If a method fails, explain and try another approach.\n"
            )
            newest_ids = self.tokenizer.encode(f"Owner: {message.strip()}\nOrbit:")
            prefix_budget = max(0, limit - len(newest_ids) - 1)
            tail_budget = max(1, limit - min(prefix_budget, len(identity_ids)))
            ids = identity_ids[:prefix_budget] + ids[-tail_budget:]
            ids = ids[-limit:]

        with self._lock:
            first_error = None
            try:
                answer = self._generate_once(ids, "greedy", on_progress)
                if answer:
                    return answer
                first_error = "The first decoding strategy returned an empty response."
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                if self.device == "cuda" and isinstance(exc, torch.cuda.OutOfMemoryError):
                    try:
                        if on_progress:
                            on_progress({
                                "kind": "method_failed",
                                "message": "GPU memory ran out. I'm switching this response to CPU.",
                                "strategy": "cuda",
                            })
                        self.model = self.model.to("cpu")
                        self.device = "cpu"
                    except Exception as move_exc:
                        first_error += f"; CPU fallback setup failed: {type(move_exc).__name__}: {move_exc}"

            if on_progress:
                on_progress({
                    "kind": "method_failed",
                    "message": f"The initial response method failed ({first_error}). I'm trying a different decoding strategy.",
                    "strategy": "greedy",
                })
                on_progress({
                    "kind": "status",
                    "message": "Switching to a controlled sampling strategy and continuing.",
                    "strategy": "sample",
                })
            try:
                answer = self._generate_once(ids, "sample", on_progress)
                if answer:
                    return answer
                raise RuntimeError("The alternate decoding strategy also returned an empty response.")
            except Exception as second_error:
                raise RuntimeError(
                    f"Both response strategies failed. Initial attempt: {first_error}. "
                    f"Alternate attempt: {type(second_error).__name__}: {second_error}"
                ) from second_error
