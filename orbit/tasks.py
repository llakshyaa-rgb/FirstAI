"""Persistent task state, progress events, and fallback execution.

TaskRunner is an execution primitive. An application must supply real callable
strategies (tools) to perform useful work; this module does not invent tools.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import sqlite3
import threading
import uuid

def _now():
    return datetime.now(timezone.utc).isoformat()

@dataclass
class TaskEvent:
    task_id: str
    kind: str
    message: str
    timestamp: str
    details: dict

class TaskStore:
    """SQLite-backed task/event store; survives process restarts."""
    def __init__(self, path="orbit_state.sqlite3"):
        self.path = str(path)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                kind TEXT NOT NULL, message TEXT NOT NULL, timestamp TEXT NOT NULL,
                details TEXT NOT NULL
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def create(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Task goal must be a non-empty string")
        task_id, now = uuid.uuid4().hex, _now()
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO tasks VALUES (?, ?, 'queued', NULL, NULL, ?, ?)",
                       (task_id, goal.strip(), now, now))
        self.emit(task_id, "queued", "Task accepted and queued.")
        return task_id

    def update(self, task_id, status, result=None, error=None):
        allowed = {"queued", "running", "completed", "failed", "needs_input", "interrupted"}
        if status not in allowed:
            raise ValueError(f"Invalid task status: {status}")
        with self._lock, self._connect() as db:
            cur = db.execute("UPDATE tasks SET status=?, result=?, error=?, updated_at=? WHERE id=?",
                             (status, result, error, _now(), task_id))
            if cur.rowcount != 1:
                raise KeyError(task_id)

    def emit(self, task_id, kind, message, **details):
        event = TaskEvent(task_id, kind, str(message), _now(), details)
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO events(task_id, kind, message, timestamp, details) VALUES (?, ?, ?, ?, ?)",
                       (task_id, kind, event.message, event.timestamp, json.dumps(details, ensure_ascii=False)))
        return event

    def get(self, task_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def events(self, task_id, after=0):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM events WHERE task_id=? AND seq>? ORDER BY seq",
                              (task_id, int(after))).fetchall()
        return [dict(row) | {"details": json.loads(row["details"])} for row in rows]

    def recent_history(self, limit=8):
        """Return completed chat turns in chronological order for conversational context."""
        if limit < 1:
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT goal, result FROM tasks WHERE status='completed' AND result IS NOT NULL ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["goal"]})
            history.append({"role": "assistant", "content": row["result"]})
        return history

    def recover_interrupted(self):
        """Mark queued/running tasks left by a dead process; the application may safely retry them."""
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT id FROM tasks WHERE status IN ('running', 'queued')").fetchall()
            ids = [r["id"] for r in rows]
            db.execute("UPDATE tasks SET status='interrupted', updated_at=? WHERE status IN ('running', 'queued')", (_now(),))
        for task_id in ids:
            self.emit(task_id, "interrupted", "Execution stopped unexpectedly; task state is preserved for recovery.")
        return ids

class TaskRunner:
    """Runs ordered strategies and continues after recoverable method failures."""
    def __init__(self, store, progress_callback=None):
        self.store = store
        self.progress_callback = progress_callback

    def _emit(self, task_id, kind, message, **details):
        event = self.store.emit(task_id, kind, message, **details)
        if self.progress_callback:
            self.progress_callback(event)
        return event

    def run(self, task_id, strategies):
        """strategies: iterable of (human-readable name, callable(goal) -> result)."""
        task = self.store.get(task_id)
        if not task:
            raise KeyError(task_id)
        if not strategies:
            raise ValueError("At least one execution strategy is required")
        self.store.update(task_id, "running")
        self._emit(task_id, "started", "I'm beginning work on the requested outcome.")
        failures = []
        for name, strategy in strategies:
            self._emit(task_id, "attempt", f"I'm using {name}.", strategy=name)
            try:
                result = strategy(task["goal"])
                if result is None:
                    raise RuntimeError("Strategy returned no result")
                serialized = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                self.store.update(task_id, "completed", result=serialized)
                self._emit(task_id, "completed", "The task is complete. The result is ready.", strategy=name)
                return result
            except Exception as exc:
                failure = {"strategy": name, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                self._emit(task_id, "method_failed",
                           f"{name} did not work ({type(exc).__name__}: {exc}). I'm switching to another approach.",
                           **failure)
        summary = json.dumps(failures, ensure_ascii=False)
        self.store.update(task_id, "failed", error=summary)
        self._emit(task_id, "blocked", "I've exhausted the available approaches. The task is not complete; see the recorded failures.",
                   failures=failures)
        raise RuntimeError("All execution strategies failed: " + summary)
