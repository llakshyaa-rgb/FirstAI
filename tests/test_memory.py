import tempfile
import unittest
from pathlib import Path

from orbit.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(str(Path(self.temp.name) / "memory.sqlite3"))

    def tearDown(self):
        self.temp.cleanup()

    def test_captures_explicit_project_and_preference_statements(self):
        saved = self.memory.observe_user_message("My project is called Nebula.")
        self.assertEqual(saved, [{"key": "Current project", "value": "Nebula"}])
        saved = self.memory.observe_user_message("I prefer concise answers.")
        self.assertEqual(saved, [{"key": "Response preference", "value": "concise answers"}])
        context = self.memory.prompt_context()
        self.assertIn("Current project: Nebula", context)
        self.assertIn("Response preference: concise answers", context)

    def test_does_not_capture_ambiguous_or_sensitive_statements(self):
        self.assertEqual(self.memory.observe_user_message("Hey Orbit!"), [])
        self.assertEqual(self.memory.observe_user_message("My medication is..."), [])
        self.assertEqual(self.memory.get_all(), [])

    def test_memory_can_be_updated_and_deleted(self):
        self.memory.set("Response preference", "short answers")
        self.memory.set("Response preference", "detailed answers")
        self.assertEqual(len(self.memory.get_all()), 1)
        self.assertEqual(self.memory.get_all()[0]["value"], "detailed answers")
        self.assertTrue(self.memory.delete("Response preference"))
        self.assertFalse(self.memory.delete("Response preference"))


if __name__ == "__main__":
    unittest.main()
