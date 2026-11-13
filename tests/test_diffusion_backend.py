import unittest
from orbit.diffusion_backend import DiffusionImageBackend

class DiffusionBackendValidationTests(unittest.TestCase):
    def setUp(self):
        self.backend = DiffusionImageBackend(model_id="test/model")

    def test_rejects_empty_prompt_before_loading_model(self):
        with self.assertRaises(ValueError):
            self.backend.generate("")

    def test_rejects_oversized_dimensions_before_loading_model(self):
        with self.assertRaises(ValueError):
            self.backend.generate("A landscape", width=2048, height=2048)

    def test_rejects_invalid_step_count_before_loading_model(self):
        with self.assertRaises(ValueError):
            self.backend.generate("A landscape", steps=0)

if __name__ == "__main__":
    unittest.main()
