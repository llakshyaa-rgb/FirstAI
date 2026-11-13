import random
import tempfile
import unittest
from pathlib import Path
from firstai.bpe import BPETokenizer
from finetune_instructions import encode_examples, load_examples, make_batch


class InstructionDataTests(unittest.TestCase):
    def test_committed_starter_corpus_is_loadable(self):
        path = Path(__file__).resolve().parents[1] / "data" / "orbit_instructions.example.jsonl"
        examples = load_examples(path)
        self.assertGreaterEqual(len(examples), 40)

    def test_loss_mask_excludes_prompt_and_keeps_response(self):
        tokenizer = BPETokenizer()
        rows = encode_examples([
            ("Owner: say hello", "Hello. How can I help?"),
            ("Owner: explain memory", "Memory stores useful context."),
        ], tokenizer, block_size=96)
        random.seed(5)
        x, y = make_batch(rows[:1], batch_size=1, block_size=96, device="cpu")
        tokens, prompt_length = rows[0]
        self.assertEqual(tuple(x.shape), (1, 96))
        self.assertTrue((y[0, :max(0, prompt_length - 1)] == -100).all())
        self.assertTrue((y[0] != -100).any())


if __name__ == "__main__":
    unittest.main()
