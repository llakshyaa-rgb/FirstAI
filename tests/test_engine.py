import unittest
from firstai.bpe import BPETokenizer


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


if __name__ == "__main__":
    unittest.main()
