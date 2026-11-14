"""Connect a trained FirstGPT checkpoint to Orbit's identity and chat memory.

Identity conditioning does not make an untrained checkpoint instruction-capable.
Generation retries use a different decoding strategy; they cannot repair missing
knowledge or replace a better-trained model.
"""
from __future__ import annotations
import threading
import os
from contextlib import contextmanager
from pathlib import Path
import torch
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer
from .identity import build_compact_system_prompt


class ModelNotReady(RuntimeError):
    pass


class OrbitRuntime:
    def __init__(self, checkpoint="checkpoints/firstgpt.best.pt", device=None,
                 max_history_messages=8, max_new_tokens=120):
        self.checkpoint_path = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_messages = max_history_messages
        self.max_new_tokens = max(16, min(512, int(os.environ.get("ORBIT_MAX_NEW_TOKENS", str(max_new_tokens)))))
        self.cpu_threads = max(1, min(8, int(os.environ.get("ORBIT_CPU_THREADS", "4"))))
        try:
            torch.set_num_threads(self.cpu_threads)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.gpu_memory_fraction = min(0.90, max(0.50, float(os.environ.get("ORBIT_GPU_MEMORY_FRACTION", "0.86"))))
        self._amp_enabled = False
        self._amp_dtype = None
        self._lock = threading.Lock()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device=0)
                torch.backends.cuda.matmul.allow_tf32 = True
            except (AttributeError, RuntimeError, ValueError):
                pass
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
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self._amp_enabled = True
                self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    @contextmanager
    def offload_for_external_gpu_work(self):
        """Temporarily move FirstGPT to CPU while another model uses the GPU."""
        with self._lock:
            original_device = self.device
            was_amp = self._amp_enabled
            original_dtype = self._amp_dtype
            moved = bool(self.model is not None and original_device.startswith("cuda"))
            if moved:
                self.model.to("cpu")
                self.device = "cpu"
                self._amp_enabled = False
                self._amp_dtype = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            try:
                yield
            finally:
                if moved and self.model is not None:
                    self.model.to(original_device)
                    self.device = original_device
                    self._amp_enabled = was_amp
                    self._amp_dtype = original_dtype

    @property
    def ready(self):
        return self.model is not None and self.tokenizer is not None

    def _generate_once(self, prompt_ids, mode, on_progress=None, max_new_tokens=None):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = []
        with torch.inference_mode():
            for index in range(self.max_new_tokens if max_new_tokens is None else max(1, min(1024, int(max_new_tokens)))):
                context = input_ids[:, -self.model.config.block_size:]
                with torch.autocast(
                    device_type="cuda", dtype=self._amp_dtype, enabled=self._amp_enabled
                ):
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
                decoded_so_far = self.tokenizer.decode(generated)
                if on_progress and (index + 1) % 8 == 0:
                    partial = self.tokenizer.decode(generated).strip()
                    on_progress({
                        "kind": "partial",
                        "message": partial,
                        "tokens_generated": index + 1,
                        "strategy": mode,
                    })
        return self.tokenizer.decode(generated).strip()

    def _build_prompt_ids(self, message, history=None, owner_preferences=None):
        """Budget context deliberately: current owner request always has priority."""
        limit = self.model.config.block_size
        system = self.tokenizer.encode(build_compact_system_prompt() + "\n")
        system = system[:min(96, max(24, limit // 5))]
        prefs = []
        if owner_preferences:
            prefs = self.tokenizer.encode("Relevant owner preferences and saved context: " + str(owner_preferences)[:1800] + "\n")
            prefs = prefs[:min(180, max(0, limit // 5))]
        fixed = system + prefs
        message_ids = self.tokenizer.encode(f"Owner: {message.strip()}\nOrbit:")
        history_ids = []
        available_for_history = max(0, limit - len(fixed) - len(message_ids))
        if available_for_history:
            history_budget = min(max(24, limit // 4), available_for_history)
            for item in reversed((history or [])[-8:]):
                role = "Owner" if item.get("role") == "user" else "Orbit"
                part = self.tokenizer.encode(f"{role}: {str(item.get('content', ''))}\n")
                take = min(len(part), history_budget - len(history_ids))
                if take <= 0:
                    break
                history_ids = part[-take:] + history_ids
                if len(history_ids) >= history_budget:
                    break
        budget = max(8, limit - len(fixed) - len(history_ids))
        if len(message_ids) > budget:
            marker = self.tokenizer.encode(" ... ")
            marker = marker[:min(len(marker), max(1, budget // 10))]
            available = max(2, budget - len(marker))
            left = max(1, available // 3)
            right = available - left
            message_ids = message_ids[:left] + marker + (message_ids[-right:] if right else [])
        ids = fixed + history_ids + message_ids
        if len(ids) > limit:
            ids = ids[-limit:]
        return ids

    def _split_long_message(self, message, max_tokens):
        """Split a long owner request into ordered, overlapping token chunks without dropping its middle."""
        ids = self.tokenizer.encode(message)
        if len(ids) <= max_tokens:
            return [message]
        chunks, start = [], 0
        overlap = min(48, max(8, max_tokens // 12))
        while start < len(ids):
            end = min(len(ids), start + max_tokens)
            # Prefer a paragraph/sentence boundary near the end while keeping progress guaranteed.
            if end < len(ids):
                floor = start + max(1, max_tokens // 2)
                for candidate in range(end, floor, -1):
                    token_text = self.tokenizer.decode(ids[candidate - 1:candidate])
                    if any(mark in token_text for mark in ("\n", ".", "?", "!", ";")):
                        end = candidate
                        break
            chunks.append(self.tokenizer.decode(ids[start:end]))
            if end >= len(ids):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _reply_long_message(self, message, on_progress=None):
        """Read every chunk in order, carry forward a compact working brief, then answer the full request."""
        limit = self.model.config.block_size
        fixed = min(96, max(32, limit // 5))
        chunk_budget = max(64, limit - fixed - 96)
        chunks = self._split_long_message(message, chunk_budget)
        notes = ""
        for index, chunk in enumerate(chunks, 1):
            if on_progress:
                on_progress({"kind": "status", "message": f"Reading all parts of your long request ({index}/{len(chunks)}).", "chunk": index, "chunks": len(chunks)})
            prompt = (
                "You are processing one consecutive part of a single owner request. "
                "Extract its requirements, constraints, decisions, examples, dependencies and questions. "
                "Preserve exact names, numbers and must/never rules. Do not answer the owner yet. "
                "Merge with the previous working notes without discarding earlier requirements. "
                "Keep the notes compact but complete.\nPREVIOUS NOTES:\n" + notes[-max(300, limit * 3):] +
                "\nCURRENT PART (" + str(index) + "/" + str(len(chunks)) + "):\n" + chunk +
                "\nUPDATED WORKING NOTES:"
            )
            ids = self.tokenizer.encode(prompt)[-limit:]
            notes = self._generate_once(ids, "greedy", on_progress, max_new_tokens=min(256, max(96, self.max_new_tokens)))
            if not notes:
                raise RuntimeError(f"Could not preserve the requirements in long-request part {index}.")
        final_prompt = (
            "Answer the owner's complete request using the accumulated notes. "
            "Respect every requirement and constraint. If requirements conflict, identify the conflict instead of silently dropping one. "
            "Do not claim actions were performed unless they were.\nCOMPLETE REQUEST NOTES:\n" + notes +
            "\nOWNER'S ORIGINAL REQUEST (for exact wording and priority):\n" + message[-max(128, limit // 3):] +
            "\nRESPONSE:"
        )
        final_ids = self.tokenizer.encode(final_prompt)[-limit:]
        answer = self._generate_once(final_ids, "greedy", on_progress)
        if not answer:
            raise RuntimeError("The final response for the long request was empty.")
        return answer

    def plan_tool_call(self, message, tool_descriptions):
        """Ask FirstGPT for a constrained, one-line JSON plan; callers must validate it."""
        if not self.ready:
            return ""
        safe = [item for item in tool_descriptions
                if item.get("name") in {"calculate", "web_search", "live_data", "list_files", "read_file", "search_files"}
                and not item.get("requires_confirmation", True)]
        instruction = (
            "You are Orbit's tool planner. Decide whether a listed read-only tool is needed. "
            "Return exactly one JSON object on one line with schema "
            '{"steps":[{"tool":"tool_name","arguments":{}}]}. '
            "Use one to five steps. If no listed tool is needed, return a JSON object with an empty steps array. "
            "Never invent tools or arguments. Owner request: " + str(message)[:3000] +
            " Available tools: " + str(safe) + "\nJSON:"
        )
        ids = self.tokenizer.encode(instruction)[-self.model.config.block_size:]
        with self._lock:
            try:
                return self._generate_once(ids, "greedy")
            except Exception:
                return ""

    def reply(self, message, history=None, owner_preferences=None, on_progress=None, max_new_tokens=None):
        if not self.ready:
            raise ModelNotReady(self.load_error or "No trained checkpoint is loaded.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Message must not be empty.")
        # Preserve the entire owner request. Oversized messages are processed in ordered
        # overlapping chunks instead of silently keeping only their beginning and end.
        request_tokens = self.tokenizer.encode(message)
        fixed_tokens = min(96, max(32, self.model.config.block_size // 5))
        history_tokens = min(18, max(0, self.model.config.block_size // 12)) if history else 0
        request_budget = max(64, self.model.config.block_size - fixed_tokens - history_tokens)
        if len(request_tokens) > request_budget:
            with self._lock:
                return self._reply_long_message(message, on_progress)
        prompt_ids = self._build_prompt_ids(message, history, owner_preferences)

        with self._lock:
            first_error = None
            try:
                answer = self._generate_once(prompt_ids, "greedy", on_progress, max_new_tokens)
                if answer:
                    return answer
                first_error = "The first decoding strategy returned an empty response."
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                if self.device.startswith("cuda") and isinstance(exc, torch.cuda.OutOfMemoryError):
                    try:
                        if on_progress:
                            on_progress({
                                "kind": "method_failed",
                                "message": "GPU memory ran out. I'm moving this response to CPU and continuing.",
                                "strategy": "cuda",
                            })
                        self.model = self.model.to("cpu")
                        self.device = "cpu"
                        self._amp_enabled = False
                        self._amp_dtype = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
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
                    "message": "Switching to controlled sampling and continuing.",
                    "strategy": "sample",
                })
            try:
                answer = self._generate_once(prompt_ids, "sample", on_progress, max_new_tokens)
                if answer:
                    return answer
                raise RuntimeError("The alternate decoding strategy also returned an empty response.")
            except Exception as second_error:
                raise RuntimeError(
                    f"Both response strategies failed. Initial attempt: {first_error}. "
                    f"Alternate attempt: {type(second_error).__name__}: {second_error}"
                ) from second_error
