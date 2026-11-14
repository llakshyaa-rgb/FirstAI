import unittest
from unittest.mock import patch

from download_capability_data import SOURCES, _convert
from orbit.knowledge import _search_term
from orbit.intent import classify_request
from orbit.media import search_web_images


class CapabilityDataTests(unittest.TestCase):
    def test_registered_training_sources_have_license_and_provenance(self):
        self.assertTrue({"gsm8k", "arc", "squad", "mbpp"}.issubset(SOURCES))
        for item in SOURCES.values():
            self.assertTrue(item["license"])
            self.assertTrue(item["url"].startswith("https://"))
            self.assertEqual(item["split"], "train")

    def test_gsm8k_conversion_preserves_solution(self):
        item = _convert("gsm8k", {"question": "What is 2+2?", "answer": "4"}, 7, SOURCES["gsm8k"])
        self.assertIn("2+2", item["prompt"])
        self.assertEqual(item["response"], "4")
        self.assertEqual(item["metadata"]["source_record_index"], 7)

    def test_arc_conversion_maps_answer_label_to_text(self):
        row = {
            "question": "Which is a planet?",
            "choices": {"label": ["A", "B"], "text": ["Moon", "Mars"]},
            "answerKey": "B",
        }
        item = _convert("arc", row, 0, SOURCES["arc"])
        self.assertEqual(item["response"], "B. Mars")

    def test_squad_conversion_grounds_answer_in_passage(self):
        row = {"context": "The Earth orbits the Sun.", "question": "What orbits the Sun?",
               "answers": {"text": ["The Earth"]}}
        item = _convert("squad", row, 0, SOURCES["squad"])
        self.assertIn("The Earth orbits", item["prompt"])
        self.assertEqual(item["response"], "The Earth")

    def test_mbpp_conversion_includes_tests(self):
        row = {"text": "Add two numbers", "code": "def add(a,b): return a+b",
               "test_list": ["assert add(1,2)==3"]}
        item = _convert("mbpp", row, 0, SOURCES["mbpp"])
        self.assertIn("assert add", item["prompt"])
        self.assertIn("def add", item["response"])

    def test_knowledge_query_prefixes_are_removed(self):
        self.assertEqual(_search_term("What is quantum computing?"), "quantum computing")
        self.assertEqual(_search_term("Who is Ada Lovelace?"), "Ada Lovelace")

    def test_automatic_routing_selects_knowledge_and_live_data(self):
        self.assertIn("knowledge", classify_request("Who is Ada Lovelace?")["capabilities"])
        self.assertIn("knowledge", classify_request("Research papers about transformers")["capabilities"])
        self.assertIn("live_data", classify_request("weather in Patna")["capabilities"])

    @patch.dict("os.environ", {}, clear=True)
    @patch("orbit.media._search_commons_images", return_value=[{"source": "Wikimedia Commons"}])
    def test_safe_image_search_falls_back_to_commons_without_brave_key(self, commons):
        self.assertEqual(search_web_images("solar system", safesearch="strict")[0]["source"], "Wikimedia Commons")
        commons.assert_called_once_with("solar system", 12)


if __name__ == "__main__":
    unittest.main()
