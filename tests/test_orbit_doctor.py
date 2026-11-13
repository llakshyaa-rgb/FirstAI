import os
import tempfile
import unittest
from unittest.mock import patch
from orbit.doctor import diagnose

class DoctorTests(unittest.TestCase):
    def test_diagnostics_do_not_require_optional_services_or_leak_keys(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "BRAVE_SEARCH_API_KEY": "secret-test-value",
            "ORBIT_IMAGE_WORKFLOW": "",
            "ORBIT_VISION_MODEL": ""
        }, clear=False):
            results = diagnose(os.path.join(folder, "missing.pt"))
        self.assertTrue(any(item["name"] == "Language checkpoint" and not item["ok"] for item in results))
        self.assertFalse(any("secret-test-value" in item["detail"] for item in results))
        self.assertTrue(any(item["name"] == "Playwright browser" for item in results))

if __name__ == "__main__":
    unittest.main()
