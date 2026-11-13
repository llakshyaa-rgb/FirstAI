import unittest
from orbit.agent import LocalAgent

class BrowserCommandTests(unittest.TestCase):
    def setUp(self):
        self.agent = LocalAgent()

    def test_open_website_command_routes_to_browser(self):
        self.agent.registry._tools["browser_open"].handler = lambda url: {
            "title": "Example", "url": url, "status": 200, "text": "Visible page text"
        }
        result = self.agent.try_execute("open website: https://example.com")
        self.assertIn("Visible page text", result)
        self.assertIn("HTTP status: 200", result)

    def test_fill_command_does_not_submit_form(self):
        self.agent.registry._tools["browser_fill"].handler = lambda selector, value: {
            "selector": selector, "filled": True
        }
        result = self.agent.try_execute("fill website: input[name=q]\nsearch phrase")
        self.assertIn("Nothing was submitted", result)

    def test_click_command_uses_owner_selected_selector(self):
        self.agent.registry._tools["browser_click"].handler = lambda selector: {
            "url": "https://example.com/next", "text": "Next page"
        }
        result = self.agent.try_execute("click website: text=Continue")
        self.assertIn("Next page", result)

if __name__ == "__main__":
    unittest.main()
