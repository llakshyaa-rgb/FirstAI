import tempfile
import io
import tarfile
import unittest
import zipfile
from pathlib import Path
from orbit.media import extract_text

class CommonMediaFormatTests(unittest.TestCase):
    def test_html_extracts_visible_text_and_skips_scripts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "page.html"
            path.write_text("<h1>Hello</h1><script>secret()</script><p>World</p>", encoding="utf-8")
            result = extract_text(path)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertNotIn("secret", result)

    def test_rtf_extracts_plain_words(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "note.rtf"
            path.write_text(r"{\rtf1\ansi Hello \b Orbit\b0}", encoding="utf-8")
            result = extract_text(path)
        self.assertIn("Hello", result)
        self.assertIn("Orbit", result)

    def test_odt_extracts_document_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "note.odt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("content.xml", '<document><p>OpenDocument text</p></document>')
            result = extract_text(path)
        self.assertIn("OpenDocument text", result)

    def test_tar_archive_can_be_inspected_without_extracting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.tar.gz"
            payload = b"safe source file contents"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo("src/main.py")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            from orbit.media_processing import _process_tar
            result = _process_tar(path)
        self.assertIn("safe source file contents", result)

    def test_epub_extracts_chapter_text_without_script(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "book.epub"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("OEBPS/chapter.xhtml", '<html><body><h1>Chapter One</h1><script>tracking()</script></body></html>')
            result = extract_text(path)
        self.assertIn("Chapter One", result)
        self.assertNotIn("tracking", result)

if __name__ == "__main__":
    unittest.main()
