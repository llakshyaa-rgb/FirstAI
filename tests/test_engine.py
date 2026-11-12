import tempfile
import unittest
from pathlib import Path

from firstai.bpe import BPETokenizer
from prepare_corpus import collect_documents


class BPETests(unittest.TestCase):
    def test_round_trip_utf8(self):
        text = "Hello, world! नमस्ते 🌍\nAI learns patterns."
        tokenizer = BPETokenizer.train(text * 20, vocab_size=320)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unknown_characters_are_supported(self):
        tokenizer = BPETokenizer.train("abc abc abc " * 20, vocab_size=280)
        text = "A new 🧠 character"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_is_bounded(self):
        tokenizer = BPETokenizer.train("ababababcdcdcd" * 10, vocab_size=260)
        self.assertLessEqual(tokenizer.vocab_size, 260)
        self.assertGreaterEqual(tokenizer.vocab_size, 256)


class CorpusTests(unittest.TestCase):
    def test_duplicate_paragraphs_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
            (root / "b.txt").write_text("Second paragraph.\n\nThird paragraph.", encoding="utf-8")
            corpus, stats = collect_documents(root)
            self.assertEqual(corpus.count("Second paragraph."), 1)
            self.assertEqual(stats["duplicates"], 1)
            self.assertEqual(stats["kept"], 3)


if __name__ == "__main__":
    unittest.main()
