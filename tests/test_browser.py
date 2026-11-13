import unittest
from orbit.browser import validate_public_url

class BrowserValidationTests(unittest.TestCase):
    def test_accepts_public_http_and_https_addresses(self):
        self.assertEqual(validate_public_url("https://8.8.8.8/"), "https://8.8.8.8/")
        self.assertEqual(validate_public_url("http://1.1.1.1"), "http://1.1.1.1")

    def test_rejects_local_and_private_hosts(self):
        for url in ("http://localhost", "http://127.0.0.1", "http://192.168.1.2",
                    "http://10.0.0.2", "http://[::1]", "http://service.local"):
            with self.subTest(url=url):
                with self.assertRaises((ValueError, PermissionError)):
                    validate_public_url(url)

    def test_rejects_non_web_schemes_and_missing_hosts(self):
        for url in ("file:///etc/passwd", "javascript:alert(1)", "https:///missing"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_public_url(url)

if __name__ == "__main__":
    unittest.main()
