import unittest
from firstai.bpe import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_round_trip_ascii_and_unicode(self):
        text = "FirstGPT learns byte pairs. नमस्ते 🌍"
        tokenizer = BPETokenizer.train((text + "\n") * 20, vocab_size=300,
                                       min_frequency=2)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_checkpoint_serialization_preserves_encoding(self):
        text = "repeatable tokenizer serialization"
        tokenizer = BPETokenizer.train((text + "\n") * 20, vocab_size=300)
        restored = BPETokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)

    def test_unknown_unicode_is_still_encodable(self):
        tokenizer = BPETokenizer.train("hello world " * 20, vocab_size=280)
        text = "🛰️漢字"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
