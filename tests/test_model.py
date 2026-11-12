import unittest
import torch
from firstai.model import FirstGPT, GPTConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = GPTConfig(vocab_size=32, block_size=16, n_layer=2,
                                n_head=2, n_embd=16, dropout=0.0)
        self.model = FirstGPT(self.config)

    def test_forward_shape_and_loss(self):
        tokens = torch.randint(0, 32, (3, 8))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (3, 8, 32))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_future_tokens_cannot_change_past_logits(self):
        self.model.eval()
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 3, 9, 8]])
        logits_a, _ = self.model(a)
        logits_b, _ = self.model(b)
        self.assertTrue(torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6))

    def test_gradients_are_finite(self):
        tokens = torch.randint(0, 32, (2, 8))
        _, loss = self.model(tokens, tokens)
        loss.backward()
        grads = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))

    def test_generation_extends_prompt(self):
        self.model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        result = self.model.generate(prompt, max_new_tokens=5, top_k=5)
        self.assertEqual(tuple(result.shape), (1, 8))
        self.assertTrue(torch.equal(result[:, :3], prompt))

    def test_invalid_head_configuration_fails(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=10, n_embd=15, n_head=4)


if __name__ == '__main__':
    unittest.main()
