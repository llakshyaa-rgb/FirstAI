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

    def test_priority_queue_matches_reference_merge_order(self):
        import random
        rng = random.Random(17)
        training = " ".join(
            "".join(rng.choice("abcde xyz") for _ in range(40))
            for _ in range(100)
        )
        tokenizer = BPETokenizer.train(training, vocab_size=320)

        def reference(text):
            tokens = list(text.encode("utf-8"))
            while len(tokens) > 1:
                ranked = [(tokenizer.ranks[pair], pair)
                          for pair in zip(tokens, tokens[1:])
                          if pair in tokenizer.ranks]
                if not ranked:
                    break
                _, best = min(ranked, key=lambda item: item[0])
                merged, index = [], 0
                while index < len(tokens):
                    if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == best:
                        merged.append(256 + tokenizer.ranks[best])
                        index += 2
                    else:
                        merged.append(tokens[index])
                        index += 1
                tokens = merged
            return tokens

        for _ in range(30):
            sample = "".join(rng.choice("abcde xyz") for _ in range(60))
            self.assertEqual(tokenizer.encode(sample), reference(sample))

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
