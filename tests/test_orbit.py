import tempfile
import unittest
from pathlib import Path
from orbit.identity import ORBIT_IDENTITY, build_system_prompt
from orbit.tasks import TaskStore, TaskRunner

class OrbitIdentityTests(unittest.TestCase):
    def test_identity_contains_jarvis_style_and_recovery(self):
        prompt = build_system_prompt()
        self.assertIn("Jarvis-inspired", prompt)
        self.assertIn("continue", prompt.lower())
        self.assertIn("method fails", prompt.lower())
        self.assertEqual(ORBIT_IDENTITY["name"], "Orbit")

class OrbitTaskTests(unittest.TestCase):
    def test_fallback_continues_and_persists_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "state.db")
            task_id = store.create("build a demo")
            attempts = []
            def broken(goal):
                attempts.append("first")
                raise RuntimeError("temporary failure")
            def working(goal):
                attempts.append("second")
                return {"status": "built", "goal": goal}
            events = []
            result = TaskRunner(store, events.append).run(
                task_id, [("primary method", broken), ("fallback method", working)]
            )
            self.assertEqual(result["status"], "built")
            self.assertEqual(attempts, ["first", "second"])
            self.assertEqual(store.get(task_id)["status"], "completed")
            self.assertIn("method_failed", [e.kind for e in events])
            self.assertEqual(len(store.events(task_id)), 6)

    def test_all_failures_are_reported_and_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "state.db")
            task_id = store.create("do work")
            def broken(_):
                raise ValueError("no route")
            with self.assertRaises(RuntimeError):
                TaskRunner(store).run(task_id, [("route A", broken), ("route B", broken)])
            self.assertEqual(store.get(task_id)["status"], "failed")

    def test_recent_history_returns_completed_turns_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "state.db")
            first = store.create("Hello")
            store.update(first, "completed", result="Good evening.")
            second = store.create("How are you?")
            store.update(second, "completed", result="Operating normally.")
            history = store.recent_history(limit=2)
            self.assertEqual([item["role"] for item in history],
                             ["user", "assistant", "user", "assistant"])
            self.assertEqual(history[-1]["content"], "Operating normally.")

    def test_interrupted_running_task_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "state.db")
            task_id = store.create("continue later")
            store.update(task_id, "running")
            self.assertEqual(store.recover_interrupted(), [task_id])
            self.assertEqual(store.get(task_id)["status"], "interrupted")

if __name__ == "__main__":
    unittest.main()
