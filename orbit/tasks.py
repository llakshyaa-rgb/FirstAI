"""Persistent task state, progress events, and fallback execution.

TaskRunner is an execution primitive. An application must supply real callable
strategies (tools) to perform useful work; this module does not invent tools.
"""
from __future__ import annotations
from dataclasses import dataclass
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
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                kind TEXT NOT NULL, message TEXT NOT NULL, timestamp TEXT NOT NULL,
                details TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(tasks)").fetchall()}
            if "conversation_id" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN conversation_id TEXT")
            # Preserve legacy chats by giving every old chat task its own conversation.
            legacy = db.execute(
                "SELECT id, goal, created_at, updated_at FROM tasks WHERE conversation_id IS NULL "
                "AND goal NOT LIKE 'Generate image: %' AND goal NOT LIKE 'Design %'"
            ).fetchall()
            for row in legacy:
                db.execute(
                    "INSERT OR IGNORE INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                    (row["id"], row["goal"][:90], row["created_at"], row["updated_at"])
                )
                db.execute("UPDATE tasks SET conversation_id=? WHERE id=?", (row["id"], row["id"]))

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.row_factory = sqlite3.Row
        return db

    def create_conversation(self, title="New chat"):
        conversation_id, now = uuid.uuid4().hex, _now()
        clean_title = str(title or "New chat").strip()[:90] or "New chat"
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (conversation_id, clean_title, now, now)
            )
        return conversation_id

    def has_conversation(self, conversation_id):
        if not conversation_id:
            return False
        with self._connect() as db:
            return db.execute("SELECT 1 FROM conversations WHERE id=?", (str(conversation_id),)).fetchone() is not None

    def create(self, goal, conversation_id=None):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Task goal must be a non-empty string")
        task_id, now = uuid.uuid4().hex, _now()
        if not conversation_id:
            conversation_id = self.create_conversation(goal[:90])
        with self._lock, self._connect() as db:
            row = db.execute("SELECT title FROM conversations WHERE id=?", (str(conversation_id),)).fetchone()
            if not row:
                raise ValueError("Conversation not found. Start a new chat and try again.")
            if row["title"] == "New chat":
                db.execute("UPDATE conversations SET title=? WHERE id=?", (goal.strip()[:90], conversation_id))
            db.execute(
                "INSERT INTO tasks(id,goal,status,result,error,created_at,updated_at,conversation_id) "
                "VALUES (?, ?, 'queued', NULL, NULL, ?, ?, ?)",
                (task_id, goal.strip(), now, now, str(conversation_id))
            )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, str(conversation_id)))
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

    def recent_history(self, limit=8, conversation_id=None):
        """Return only the selected conversation's completed turns, in chronological order."""
        if limit < 1:
            return []
        with self._connect() as db:
            if conversation_id:
                rows = db.execute(
                    "SELECT goal, result FROM tasks WHERE status='completed' AND result IS NOT NULL "
                    "AND conversation_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (str(conversation_id), int(limit))
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT goal, result FROM tasks WHERE status='completed' AND result IS NOT NULL "
                    "AND goal NOT LIKE 'Generate image: %' AND goal NOT LIKE 'Design %' "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (int(limit),)
                ).fetchall()
        history = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["goal"]})
            history.append({"role": "assistant", "content": row["result"]})
        return history

    def recent_conversations(self, limit=60):
        """Return saved conversations, not individual message tasks."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.id,c.title,c.created_at,c.updated_at,"
                "(SELECT t.status FROM tasks t WHERE t.conversation_id=c.id ORDER BY t.created_at DESC,t.rowid DESC LIMIT 1) AS status "
                "FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
                (max(1, min(200, int(limit))),)
            ).fetchall()
        return [dict(row) for row in rows]

    def conversation(self, conversation_id):
        """Return every saved user/assistant turn in one conversation."""
        with self._connect() as db:
            conv = db.execute(
                "SELECT id,title,created_at,updated_at FROM conversations WHERE id=?",
                (str(conversation_id),)
            ).fetchone()
            if not conv:
                return None
            rows = db.execute(
                "SELECT id,goal,result,status,created_at,updated_at FROM tasks "
                "WHERE conversation_id=? ORDER BY created_at,rowid",
                (str(conversation_id),)
            ).fetchall()
        messages = []
        for row in rows:
            messages.append({"role": "user", "content": row["goal"], "task_id": row["id"], "status": row["status"], "timestamp": row["created_at"]})
            if row["result"]:
                messages.append({"role": "assistant", "content": row["result"], "task_id": row["id"], "status": row["status"], "timestamp": row["updated_at"]})
        return dict(conv) | {"messages": messages}

    def recent_images(self, limit=20):
        """Return completed image artifacts for restoring the local conversation view."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, goal, updated_at FROM tasks WHERE status='completed' AND goal LIKE 'Generate image: %' ORDER BY updated_at DESC LIMIT ?",
                (int(limit),)
            ).fetchall()
        images = []
        for row in rows:
            for event in self.events(row["id"]):
                if event["kind"] == "image" and event["details"].get("url"):
                    images.append({
                        "task_id": row["id"],
                        "prompt": row["goal"].removeprefix("Generate image: "),
                        "url": event["details"]["url"],
                        "timestamp": event["timestamp"],
                    })
        return images

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
