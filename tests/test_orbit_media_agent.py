import unittest
from unittest.mock import patch
from orbit.agents import AgentTeam

class StubRegistry:
    def describe(self):
        return []

class MediaSpecialistTests(unittest.TestCase):
    def test_coordinator_assigns_media_agent_for_attachments(self):
        with patch("orbit.agents.process_upload", return_value={
            "name": "notes.txt", "kind": "document", "content": "important details"
        }):
            results = AgentTeam(StubRegistry()).run("summarise this", attachments=["fake-id"])
        self.assertEqual(results[0]["agent"], "media")
        self.assertTrue(results[0]["ok"])
        self.assertIn("important details", results[0]["result"][0]["content"])

    def test_media_agent_uses_extracted_text_when_processing_fails(self):
        with patch("orbit.agents.process_upload", side_effect=RuntimeError("vision offline")), \
             patch("orbit.agents.get_upload", return_value={"text": "fallback document text"}):
            results = AgentTeam(StubRegistry()).run("analyse this", attachments=["fake-id"])
        self.assertFalse(results[0]["result"][0]["ok"])
        self.assertEqual(results[0]["result"][0]["fallback_text"], "fallback document text")

if __name__ == "__main__":
    unittest.main()
