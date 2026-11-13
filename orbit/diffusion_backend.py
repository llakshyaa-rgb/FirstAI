"""Optional in-process Diffusers image backend.

The language model and image model share one Python process and CUDA allocator.
Orbit serializes generation jobs; callers should offload the language model while
rendering and restore it afterwards.
"""
from __future__ import annotations
import io
import os
import threading


class DiffusionImageBackend:
    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or os.environ.get("ORBIT_IMAGE_MODEL", "Lykon/dreamshaper-8")
        self.device = device or os.environ.get("ORBIT_IMAGE_DEVICE", "auto")
        self._pipeline = None
        self._lock = threading.RLock()

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-image.txt to enable the Diffusers image backend.") from exc
        use_cuda = self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ORBIT_IMAGE_DEVICE=cuda was selected, but CUDA is unavailable.")
        dtype = torch.float16 if use_cuda else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, use_safetensors=True
        )
        if use_cuda:
            try:
                pipeline.enable_attention_slicing()
                pipeline.enable_vae_slicing()
                pipeline.enable_model_cpu_offload()
            except (AttributeError, ImportError):
                pipeline.to("cuda")
        else:
            pipeline.to("cpu")
        self._pipeline = pipeline
        return pipeline

    def health(self):
        return {"backend": "diffusers", "model": self.model_id, "loaded": self._pipeline is not None}

    def generate(self, prompt, negative_prompt="", seed=None, width=512, height=512,
                 steps=20, guidance=7.0, on_progress=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
            raise ValueError("Image prompt must contain 1-5000 characters.")
        if not isinstance(negative_prompt, str) or len(negative_prompt) > 2000:
            raise ValueError("Negative prompt must be at most 2000 characters.")
        if type(width) is not int or type(height) is not int or not (256 <= width <= 768 and 256 <= height <= 768):
            raise ValueError("Image dimensions must be integers between 256 and 768.")
        if width % 8 or height % 8:
            raise ValueError("Image dimensions must be divisible by 8.")
        if type(steps) is not int or not 1 <= steps <= 60:
            raise ValueError("Image steps must be between 1 and 60.")
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not 0 <= float(guidance) <= 20:
            raise ValueError("Guidance must be between 0 and 20.")
        with self._lock:
            import torch
            pipeline = self._load()
            generator = None
            if seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(int(seed))
            if on_progress:
                on_progress({"kind": "status", "message": "The image model is rendering the requested artwork."})
            result = pipeline(
                prompt=prompt.strip(), negative_prompt=negative_prompt.strip() or None,
                width=width, height=height, num_inference_steps=steps,
                guidance_scale=float(guidance), generator=generator,
            )
            if not result.images:
                raise RuntimeError("Diffusers completed without returning an image.")
            stream = io.BytesIO()
            result.images[0].save(stream, format="PNG")
            image_bytes = stream.getvalue()
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [{"info": {"filename": "orbit-generated.png"}, "content": image_bytes}]

    def unload(self):
        with self._lock:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
