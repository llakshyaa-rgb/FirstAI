import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from orbit.media_processing import _process_zip, process_upload

class MediaProcessingTests(unittest.TestCase):
    def test_zip_text_files_are_read_without_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("notes/readme.md", "Orbit archive content")
                archive.writestr("run.sh", "echo should not be parsed")
            result = _process_zip(path)
            self.assertIn("Orbit archive content", result)
            self.assertNotIn("should not be parsed", result)

    def test_image_processing_passes_pixels_to_vision_adapter(self):
        item = {"name": "picture.png", "stored_name": "a" * 32 + ".png", "category": "image", "text": ""}
        with patch("orbit.media_processing.get_upload", return_value=item), patch("orbit.media_processing.describe_image", return_value="A cat on a sofa.") as vision:
            result = process_upload("a" * 32, "What is in this picture?")
        self.assertEqual(result["content"], "A cat on a sofa.")
        vision.assert_called_once_with("a" * 32, "What is in this picture?")

    def test_unknown_format_is_reported_as_opaque(self):
        item = {"name": "sample.unknown", "stored_name": "sample.unknown", "category": "file", "text": ""}
        with patch("orbit.media_processing.get_upload", return_value=item), patch("orbit.media_processing.UPLOAD_DIR", Path(".")):
            result = process_upload("a" * 32)
        self.assertEqual(result["kind"], "file")
        self.assertIn("no content parser", result["content"])

if __name__ == "__main__":
    unittest.main()
