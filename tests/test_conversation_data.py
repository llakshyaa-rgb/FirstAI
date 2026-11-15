import unittest

from download_conversation_data import _messages, convert_conversation


class ConversationDataTests(unittest.TestCase):
    def test_message_roles_are_normalized(self):
        rows = _messages({"messages": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
        ]})
        self.assertEqual(rows, [
            ("user", "Hi."),
            ("assistant", "Hello."),
        ])

    def test_each_assistant_turn_keeps_prior_turn_context(self):
        rows = list(convert_conversation([
            ("user", "My project is called Nebula."),
            ("assistant", "Got it — Nebula."),
            ("user", "What is it called?"),
            ("assistant", "Your project is called Nebula."),
        ], 7))
        self.assertEqual(len(rows), 2)
        self.assertIn("Owner: My project is called Nebula.", rows[0]["prompt"])
        self.assertIn("Owner: What is it called?", rows[1]["prompt"])
        self.assertIn("Orbit: Got it — Nebula.", rows[1]["prompt"])
        self.assertFalse(rows[1]["prompt"].endswith("\\nOrbit:"))
        self.assertEqual(rows[1]["response"], "Orbit: Your project is called Nebula.")
        self.assertEqual(rows[1]["metadata"]["source_record_index"], 7)

    def test_long_assistant_response_is_not_a_target(self):
        rows = list(convert_conversation([
            ("user", "Explain."),
            ("assistant", "x" * 1801),
        ], 1))
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
