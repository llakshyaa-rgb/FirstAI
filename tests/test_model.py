import unittest
import torch
from firstai.model import FirstGPT, GPTConfig


class ModelTests(unittest.TestCase):
    def _exercise_architecture(self, version):
        torch.manual_seed(7)
        config = GPTConfig(vocab_size=64, block_size=16, n_layer=2,
                           n_head=2, n_embd=16, dropout=0.0,
                           architecture_version=version)
        model = FirstGPT(config)
        tokens = torch.randint(0, 64, (3, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (3, 8, 64))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))

        model.eval()
        left = torch.tensor([[1, 2, 3, 4, 5]])
        right = torch.tensor([[1, 2, 3, 9, 8]])
        logits_left, _ = model(left)
        logits_right, _ = model(right)
        self.assertTrue(torch.allclose(logits_left[:, :3],
                                       logits_right[:, :3], atol=1e-6))

        generated = model.generate(torch.tensor([[1, 2, 3]]),
                                   max_new_tokens=5, top_k=8,
                                   top_p=0.9, repetition_penalty=1.1)
        self.assertEqual(tuple(generated.shape), (1, 8))

        restored = FirstGPT(GPTConfig(**vars(config)))
        restored.load_state_dict(model.state_dict())
        restored.eval()
        restored_logits, _ = restored(left)
        self.assertTrue(torch.allclose(logits_left, restored_logits, atol=1e-6))

    def test_v1_forward_causality_gradients_generation_and_reload(self):
        self._exercise_architecture(1)

    def test_v2_forward_causality_gradients_generation_and_reload(self):
        self._exercise_architecture(2)

    def test_invalid_head_configuration_fails(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=10, n_embd=15, n_head=4)

    def test_v2_requires_even_rotary_head_dimension(self):
        with self.assertRaises(ValueError):
            GPTConfig(vocab_size=32, n_embd=18, n_head=3,
                      architecture_version=2)


if __name__ == "__main__":
    unittest.main()
