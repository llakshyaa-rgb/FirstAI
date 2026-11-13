import json
import tempfile
import unittest
from pathlib import Path
from orbit.image_workflow import load_workflow_template
from orbit.images import ComfyUIImageBackend, ImageBackendError
from unittest.mock import patch

class ImageWorkflowTests(unittest.TestCase):
    def test_prompt_negative_prompt_and_seed_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps({
                "1": {"inputs": {"text": "{{PROMPT}}"}},
                "2": {"inputs": {"text": "{{NEGATIVE_PROMPT}}"}},
                "3": {"inputs": {"seed": "{{SEED}}", "width": "{{WIDTH}}",
                                "height": "{{HEIGHT}}", "steps": "{{STEPS}}",
                                "cfg": "{{CFG}}", "checkpoint": "{{CHECKPOINT}}"}},
            }), encoding="utf-8")
            result = load_workflow_template(path, "adult character concept", "blur", 42)
            self.assertEqual(result["1"]["inputs"]["text"], "adult character concept")
            self.assertEqual(result["2"]["inputs"]["text"], "blur")
            self.assertEqual(result["3"]["inputs"]["seed"], 42)
            self.assertEqual(result["3"]["inputs"]["width"], 768)
            self.assertEqual(result["3"]["inputs"]["height"], 768)
            self.assertEqual(result["3"]["inputs"]["steps"], 24)
            self.assertEqual(result["3"]["inputs"]["cfg"], 7.0)
            self.assertEqual(result["3"]["inputs"]["checkpoint"], "model.safetensors")

    def test_completed_job_without_outputs_fails_immediately(self):
        backend = ComfyUIImageBackend()
        backend._request = lambda path: (
            json.dumps({"job": {"status": {"completed": True}, "outputs": {}}}).encode(),
            "application/json",
        )
        with self.assertRaises(ImageBackendError):
            backend.wait("job", poll_interval=0.01, timeout=1)

    def test_backend_execution_error_is_reported(self):
        backend = ComfyUIImageBackend()
        backend._request = lambda path: (
            json.dumps({"job": {"status": {
                "status_str": "error",
                "messages": [["execution_error", {"exception_message": "bad graph"}]],
            }, "outputs": {}}}).encode(),
            "application/json",
        )
        with self.assertRaises(ImageBackendError):
            backend.wait("job", poll_interval=0.01, timeout=1)

if __name__ == "__main__":
    unittest.main()
