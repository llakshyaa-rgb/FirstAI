import tempfile
import unittest
from pathlib import Path
from orbit.memory import MemoryStore
from orbit.tools import ToolRegistry
from orbit.agent import LocalAgent
from unittest.mock import patch

class MemoryTests(unittest.TestCase):
    def test_memory_persists_and_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"memory.db"
            store=MemoryStore(path);store.set("tone","concise")
            self.assertEqual(store.get_all()[0]["value"],"concise")
            self.assertTrue(store.delete("tone"))
            self.assertEqual(store.get_all(),[])
    def test_memory_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/"memory.db")
            with self.assertRaises(ValueError):store.set("", "x")
            with self.assertRaises(ValueError):store.set("x", "a"*4001)

class AgentFallbackTests(unittest.TestCase):
    def test_web_search_failure_falls_back_without_ending_task(self):
        events = []
        with patch("orbit.agent.search_web", side_effect=RuntimeError("missing key")):
            agent = LocalAgent(progress=lambda message, **details: events.append((message, details)))
            result = agent.try_execute("search web for recent transformer research")
        self.assertIsNone(result)
        self.assertTrue(any(item[1].get("kind") == "method_failed" for item in events))

class ToolTests(unittest.TestCase):
    def test_confirmation_gate(self):
        tools=ToolRegistry();tools.register("publish","Publish item",lambda item:item)
        with self.assertRaises(PermissionError):tools.call("publish",{"item":"draft"})
        self.assertEqual(tools.call("publish",{"item":"draft"},confirmed=True),"draft")
    def test_unknown_tool_rejected(self):
        with self.assertRaises(KeyError):ToolRegistry().call("missing")
if __name__=="__main__":unittest.main()
