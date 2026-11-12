import unittest
import torch
from firstai.data import get_batch, make_splits
from firstai.bpe import BPETokenizer


class DataTests(unittest.TestCase):
    def test_batches_are_shifted_by_one(self):
        source = torch.arange(40)
        x, y = get_batch(source, batch_size=8, block_size=6, device="cpu")
        self.assertEqual(tuple(x.shape), (8, 6))
        self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
        self.assertTrue(torch.equal(y[:, -1], x[:, -1] + 1))

    def test_split_preserves_all_tokens(self):
        tokenizer = BPETokenizer()
        train, validation = make_splits("abcdefghijklmnopqrstuvwxyz" * 2, tokenizer)
        self.assertEqual(len(train) + len(validation), 52)

    def test_invalid_split_fraction_fails(self):
        with self.assertRaises(ValueError):
            make_splits("abcdefghijklmnopqrstuvwxyz", BPETokenizer(), 1.0)


if __name__ == "__main__":
    unittest.main()
