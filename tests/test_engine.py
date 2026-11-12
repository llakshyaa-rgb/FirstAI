import math
import unittest

from firstai.tensor import Tensor, layer_norm, softmax_rows


class TensorEngineTests(unittest.TestCase):
    def test_matrix_multiplication(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual(a.matmul(b).tolist(), [[19.0, 22.0], [43.0, 50.0]])

    def test_autograd_through_matmul_and_mean(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 0], [0, 2]])
        loss = a.matmul(b).mean()
        loss.backward()
        self.assertAlmostEqual(a.grad[0][0], 0.5)
        self.assertAlmostEqual(b.grad[0][0], 1.0)

    def test_softmax_rows_sum_to_one(self):
        p = softmax_rows(Tensor([[1, 2, 3], [1000, 1001, 1002]]))
        for row in p.data:
            self.assertAlmostEqual(sum(row), 1.0)
        self.assertTrue(all(math.isfinite(v) for row in p.data for v in row))

    def test_layer_norm_rows_have_zero_mean(self):
        out = layer_norm(Tensor([[1, 2, 3], [10, 10, 10]]))
        for row in out.data:
            self.assertAlmostEqual(sum(row) / len(row), 0.0, places=6)

    def test_scalar_backward_guard(self):
        with self.assertRaises(ValueError):
            Tensor([[1, 2]]).backward()


if __name__ == "__main__":
    unittest.main()
