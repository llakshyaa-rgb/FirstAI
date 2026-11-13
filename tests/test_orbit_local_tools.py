import tempfile
import unittest
from pathlib import Path
from orbit.local_tools import WorkspaceTools, safe_calculate
from orbit.agent import LocalAgent
from orbit.memory import MemoryStore


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_calculate("(12 + 8) * 3 / 2"), 30)
        self.assertEqual(safe_calculate("2 ** 8"), 256)

    def test_rejects_code_and_unsafe_expressions(self):
        for expression in ("__import__('os').system('id')", "open('x')", "1 / 0", "2 ** 1000"):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, TypeError)):
                    safe_calculate(expression)


class WorkspaceTests(unittest.TestCase):
    def test_create_read_update_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = WorkspaceTools(tmp)
            tools.create_file("notes/hello.txt", "first")
            self.assertEqual(tools.read_file("notes/hello.txt"), "first")
            result = tools.update_file("notes/hello.txt", "second")
            self.assertEqual(tools.read_file("notes/hello.txt"), "second")
            self.assertTrue((Path(tmp) / result["backup"]).is_file())
            with self.assertRaises(FileExistsError):
                tools.create_file("notes/hello.txt", "third")

    def test_path_traversal_and_private_paths_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = WorkspaceTools(tmp)
            with self.assertRaises(PermissionError):
                tools.read_file("../outside.txt")
            with self.assertRaises(PermissionError):
                tools.read_file(".env")
            with self.assertRaises(PermissionError):
                tools.read_file(".git/config")
            secret = Path(tmp) / ".env"
            secret.write_text("PRIVATE=1", encoding="utf-8")
            alias = Path(tmp) / "visible-name.txt"
            alias.symlink_to(secret)
            with self.assertRaises(PermissionError):
                tools.read_file("visible-name.txt")

    def test_search_and_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = WorkspaceTools(tmp)
            tools.create_file("src/main.py", "def orbit():\n    return 'ready'\n")
            self.assertEqual(tools.search_files("orbit")[0]["path"], "src/main.py")
            self.assertTrue(any(x["name"] == "src/" for x in tools.list_files(".")))


class AgentTests(unittest.TestCase):
    def test_deterministic_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalAgent(tmp)
            self.assertEqual(agent.try_execute("calculate 6 * (7 + 1)"), "6 * (7 + 1) = 48")

    def test_create_and_update_file_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalAgent(tmp)
            created = agent.try_execute("create file: notes/story.txt\nA first draft.")
            self.assertIn("Created notes/story.txt", created)
            updated = agent.try_execute("update file: notes/story.txt\nA revised draft.")
            self.assertIn("Backup:", updated)
            self.assertEqual((Path(tmp) / "notes/story.txt").read_text(), "A revised draft.")

    def test_explicit_preferences_are_remembered_and_retrievable(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.db")
            agent = LocalAgent(tmp, memory=memory)
            answer = agent.try_execute("I prefer concise explanations.")
            self.assertIn("Noted", answer)
            saved = memory.get_all()
            self.assertEqual(len(saved), 1)
            recalled = agent.try_execute("what do you remember about me")
            self.assertIn("concise explanations", recalled)
            forgotten = agent.try_execute("forget: " + saved[0]["key"])
            self.assertEqual(forgotten, "Memory removed.")
            self.assertEqual(memory.get_all(), [])

    def test_unknown_request_is_left_for_language_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalAgent(tmp)
            self.assertIsNone(agent.try_execute("Write a thoughtful fantasy story."))


if __name__ == "__main__":
    unittest.main()
