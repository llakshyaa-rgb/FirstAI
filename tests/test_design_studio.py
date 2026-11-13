import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.design_studio import create_artifact, parse_markdown


class DesignStudioTests(unittest.TestCase):
    def test_markdown_parser_handles_headings_lists_and_tables(self):
        blocks = parse_markdown("# Title\n\nA paragraph.\n\n- One\n- Two\n\n| A | B |\n|---|---|\n| 1 | 2 |")
        kinds = [kind for kind, _ in blocks]
        self.assertEqual(kinds, ["heading", "paragraph", "bullets", "table"])
        self.assertEqual(blocks[-1][1][-1], ["1", "2"])

    def test_html_export_is_styled_and_escapes_markup(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("orbit.design_studio.OUTPUT_DIR", Path(folder)):
                item = create_artifact("A < B", "## Section\n\n**Bold** & safe", "html", theme_name="paper")
                output = (Path(folder) / item["name"]).read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", output)
        self.assertIn("&lt;", output)
        self.assertIn("<strong>Bold</strong>", output)
        self.assertIn("#9a663b", output)

    def test_csv_export_writes_table_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("orbit.design_studio.OUTPUT_DIR", Path(folder)):
                item = create_artifact("Numbers", "| Name | Score |\n|---|---|\n| Ada | 95 |", "csv")
                with (Path(folder) / item["name"]).open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.reader(stream))
        self.assertEqual(rows, [["Name", "Score"], ["Ada", "95"]])

    def test_svg_export_is_valid_vector_markup(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("orbit.design_studio.OUTPUT_DIR", Path(folder)):
                item = create_artifact("Poster", "## Main point\n\nA clear message.", "svg")
                output = (Path(folder) / item["name"]).read_text(encoding="utf-8")
        self.assertTrue(output.startswith("<svg "))
        self.assertIn("Main point", output)

    def test_json_export_preserves_structured_content(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            with patch("orbit.design_studio.OUTPUT_DIR", Path(folder)):
                item = create_artifact("Data", "## Rows\n\n| Name | Score |\n|---|---|\n| Ada | 95 |", "json")
                payload = json.loads((Path(folder) / item["name"]).read_text(encoding="utf-8"))
        self.assertEqual(payload["title"], "Data")
        self.assertEqual(payload["blocks"][-1]["content"][-1], ["Ada", "95"])
    def test_rejects_unsupported_formats_and_empty_content(self):
        with self.assertRaises(ValueError):
            create_artifact("Test", "Body", "exe")
        with self.assertRaises(ValueError):
            create_artifact("Test", "", "pdf")


if __name__ == "__main__":
    unittest.main()
