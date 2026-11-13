import tempfile
import unittest
from pathlib import Path
from orbit.local_tools import WorkspaceTools, safe_calculate
from orbit.agent import LocalAgent


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

    def test_unknown_request_is_left_for_language_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalAgent(tmp)
            self.assertIsNone(agent.try_execute("Write a thoughtful fantasy story."))


if __name__ == "__main__":
    unittest.main()
