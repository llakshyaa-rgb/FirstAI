import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from firstai.bpe import BPETokenizer
from firstai.model import FirstGPT, GPTConfig
from orbit.runtime import OrbitRuntime


class OrbitPromptTests(unittest.TestCase):
    def test_resource_limits_are_configurable_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
            "ORBIT_GPU_MEMORY_FRACTION": "0.86",
            "ORBIT_CPU_THREADS": "4",
            "ORBIT_MAX_NEW_TOKENS": "96",
        }):
            runtime = OrbitRuntime(Path(tmp) / "missing.pt", device="cpu")
            self.assertAlmostEqual(runtime.gpu_memory_fraction, 0.86)
            self.assertEqual(runtime.cpu_threads, 4)
            self.assertEqual(runtime.max_new_tokens, 96)

    def test_prompt_respects_context_and_keeps_latest_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OrbitRuntime(Path(tmp) / "missing.pt", device="cpu")
            runtime.tokenizer = BPETokenizer()
            runtime.model = FirstGPT(GPTConfig(
                vocab_size=256, block_size=96, n_layer=1, n_head=2, n_embd=16, dropout=0
            ))
            ids = runtime._build_prompt_ids(
                "Please solve this exact problem and explain the final result clearly.",
                history=[
                    {"role": "user", "content": "old question " * 30},
                    {"role": "assistant", "content": "old answer " * 30},
                ],
                owner_preferences="I prefer concise but clear explanations " * 10,
            )
            self.assertLessEqual(len(ids), 96)
            decoded = runtime.tokenizer.decode(ids)
            self.assertIn("Please solve", decoded)
            self.assertTrue(decoded.endswith("Orbit:"))

    def test_long_message_is_bounded_without_losing_its_ending(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = OrbitRuntime(Path(tmp) / "missing.pt", device="cpu")
            runtime.tokenizer = BPETokenizer()
            runtime.model = FirstGPT(GPTConfig(
                vocab_size=256, block_size=64, n_layer=1, n_head=2, n_embd=16, dropout=0
            ))
            message = "beginning " + ("unnecessary detail " * 100) + "IMPORTANT FINAL DETAIL"
            ids = runtime._build_prompt_ids(message)
            self.assertLessEqual(len(ids), 64)
            self.assertIn("IMPORTANT FINAL DETAIL", runtime.tokenizer.decode(ids))


if __name__ == "__main__":
    unittest.main()
