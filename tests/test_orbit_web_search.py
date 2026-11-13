import json
import os
import unittest
from unittest.mock import MagicMock, patch
from orbit.web_search import search_web

class WebSearchTests(unittest.TestCase):
    @patch("orbit.web_search.urllib.request.urlopen")
    def test_search_returns_source_links(self, open_url):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"web":{"results":[{
            "title":"Example result","url":"https://example.com/article",
            "description":"Useful information","meta_url":{"hostname":"example.com"}
        }]}}).encode()
        open_url.return_value = response
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY":"test-key"}):
            results = search_web("Orbit AI")
        self.assertEqual(results[0]["url"], "https://example.com/article")
        self.assertEqual(results[0]["source"], "example.com")

    def test_missing_api_key_is_reported(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "BRAVE_SEARCH_API_KEY"):
                search_web("test")

if __name__ == "__main__":
    unittest.main()
