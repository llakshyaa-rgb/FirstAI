import json
import tempfile
import unittest
from pathlib import Path
from orbit.image_workflow import load_workflow_template

class ImageWorkflowTests(unittest.TestCase):
    def test_prompt_negative_prompt_and_seed_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps({
                "1": {"inputs": {"text": "{{PROMPT}}"}},
                "2": {"inputs": {"text": "{{NEGATIVE_PROMPT}}"}},
                "3": {"inputs": {"seed": "{{SEED}}"}},
            }), encoding="utf-8")
            result = load_workflow_template(path, "adult character concept", "blur", 42)
            self.assertEqual(result["1"]["inputs"]["text"], "adult character concept")
            self.assertEqual(result["2"]["inputs"]["text"], "blur")
            self.assertEqual(result["3"]["inputs"]["seed"], "42")

if __name__ == "__main__":
    unittest.main()
