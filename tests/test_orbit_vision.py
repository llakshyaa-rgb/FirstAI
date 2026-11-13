import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from orbit.vision import describe_image

class VisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / ("a" * 32 + ".png")
        self.image.write_bytes(b"\x89PNG\r\n\x1a\nsmall-test-image")

    def tearDown(self):
        self.tmp.cleanup()

    def _response(self, payload):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(payload).encode()
        return response

    @patch("orbit.vision.urllib.request.urlopen")
    def test_ollama_receives_uploaded_image_pixels(self, open_url):
        open_url.return_value = self._response({"response": "A red car."})
        with patch("orbit.vision.UPLOAD_DIR", self.root), patch.dict(os.environ, {
            "ORBIT_VISION_PROVIDER": "ollama", "ORBIT_VISION_MODEL": "llava",
            "ORBIT_VISION_URL": "http://localhost:11434/api/generate"
        }):
            result = describe_image("a" * 32, "What is shown?")
        self.assertEqual(result, "A red car.")
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(base64.b64decode(payload["images"][0]), self.image.read_bytes())
        self.assertEqual(payload["prompt"], "What is shown?")

    @patch("orbit.vision.urllib.request.urlopen")
    def test_openai_compatible_receives_data_url(self, open_url):
        open_url.return_value = self._response({"choices": [{"message": {"content": "A landscape."}}]})
        with patch("orbit.vision.UPLOAD_DIR", self.root), patch.dict(os.environ, {
            "ORBIT_VISION_PROVIDER": "openai-compatible", "ORBIT_VISION_MODEL": "vision-model",
            "ORBIT_VISION_URL": "http://localhost:8080"
        }):
            result = describe_image("a" * 32)
        self.assertEqual(result, "A landscape.")
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

    def test_invalid_upload_id_is_rejected(self):
        with self.assertRaises(ValueError):
            describe_image("../secret")

if __name__ == "__main__":
    unittest.main()
