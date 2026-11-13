import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from orbit.media import save_upload, extract_text, parse_multipart, list_uploads

class MediaTests(unittest.TestCase):
    def test_accepts_png_and_rejects_fake_image(self):
        with tempfile.TemporaryDirectory() as tmp, patch("orbit.media.UPLOAD_DIR", Path(tmp)):
            item = save_upload("photo.png", b"\x89PNG\r\n\x1a\n" + b"data", "image/png")
            self.assertEqual(item["category"], "image")
            self.assertTrue((Path(tmp) / item["stored_name"]).is_file())
            with self.assertRaises(ValueError):
                save_upload("fake.png", b"not an image", "image/png")

    def test_unknown_file_types_are_stored_as_opaque_data(self):
        with tempfile.TemporaryDirectory() as tmp, patch("orbit.media.UPLOAD_DIR", Path(tmp)):
            item = save_upload("archive.custom", b"opaque bytes")
            self.assertEqual(item["category"], "file")
            self.assertEqual(list_uploads()[0]["name"], "archive.custom")
            self.assertEqual((Path(tmp) / item["stored_name"]).read_bytes(), b"opaque bytes")

    def test_text_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.md"
            path.write_text("# Hello\nOrbit", encoding="utf-8")
            self.assertIn("Orbit", extract_text(path))

    def test_multipart_parser(self):
        boundary = "----orbit-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"hello.txt\"\r\n"
            "Content-Type: text/plain\r\n\r\nhello orbit\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        items = parse_multipart("multipart/form-data; boundary=" + boundary, body)
        self.assertEqual(items[0][0], "hello.txt")
        self.assertEqual(items[0][1], b"hello orbit")

    def test_upload_size_limit(self):
        from orbit.media import MAX_UPLOAD_BYTES
        with self.assertRaises(ValueError):
            save_upload("large.bin", b"x" * (MAX_UPLOAD_BYTES + 1))

if __name__ == "__main__":
    unittest.main()
