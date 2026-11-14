import unittest
from unittest.mock import patch

from orbit.intent import classify_request
from orbit.live_data import _city_from_query, _rates, fetch_live_data


class LiveDataRoutingTests(unittest.TestCase):
    def test_supported_live_data_questions_route_automatically(self):
        for prompt in (
            "weather in Patna", "air quality in Delhi", "latest earthquakes",
            "USD to INR exchange rate", "India GDP", "country profile for India",
        ):
            with self.subTest(prompt=prompt):
                self.assertIn("live_data", classify_request(prompt)["capabilities"])

    def test_city_extraction_handles_common_weather_phrasing(self):
        self.assertEqual(_city_from_query("What's the weather in Patna?"), "Patna")
        self.assertEqual(_city_from_query("air quality for New Delhi"), "New Delhi")

    @patch("orbit.live_data._get_json", return_value=[{"date": "2026-11-13", "base": "INR", "quote": "USD", "rate": 0.01}])
    def test_currency_direction_follows_user_order(self, mocked_get):
        result = _rates("INR to USD exchange rate")
        self.assertIn("INR to USD", result["title"])
        self.assertIn("base=INR", result["api_url"])
        self.assertIn("quotes=USD", result["api_url"])
        self.assertTrue(result["source_url"].startswith("https://"))
        mocked_get.assert_called_once()

    @patch("orbit.live_data._earthquakes")
    def test_earthquake_query_uses_usgs_feed(self, mocked):
        mocked.return_value = {"title": "USGS feed", "records": []}
        self.assertEqual(fetch_live_data("latest earthquakes")["title"], "USGS feed")
        mocked.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
