"""Persistent, owner-controlled memory for Orbit.

Only explicit user statements are captured automatically. Memory is stored locally
in the same SQLite database as chat history and can be inspected/deleted in the UI.
"""
import re
import sqlite3
from datetime import datetime, timezone

_SENSITIVE = re.compile(
    r"\b(health|medical|diagnos|medication|religion|religious|politic|political|"
    r"sexual|sex life|orientation|race|ethnic|criminal|password|api key|secret)\b",
    re.IGNORECASE,
)

class MemoryStore:
    def __init__(self, path="orbit_state.sqlite3"):
        self.path = str(path)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("""CREATE TABLE IF NOT EXISTS memories(
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS memory_audit(
                id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,
                action TEXT NOT NULL, timestamp TEXT NOT NULL
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.row_factory = sqlite3.Row
        return db

    def set(self, key, value):
        if not isinstance(key, str) or not key.strip() or len(key) > 120:
            raise ValueError("Memory key must be 1-120 characters")
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError("Memory value must be non-empty text up to 4000 characters")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                "INSERT INTO memories(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key.strip(), value.strip(), now),
            )
            db.execute("INSERT INTO memory_audit(key,action,timestamp) VALUES(?,?,?)",
                       (key.strip(), "saved", now))

    def observe_user_message(self, message):
        """Capture only clear, non-sensitive preferences/facts explicitly stated by the owner."""
        if not isinstance(message, str):
            return []
        text = message.strip()
        if not text or len(text) > 1200 or _SENSITIVE.search(text):
            return []
        candidates = []
        patterns = [
            (r"^(?:please\s+)?remember that\s+(.{4,240})$", "Owner note"),
            (r"^(?:my\s+)?(?:current\s+)?project is called\s+(.{2,100})$", "Current project"),
            (r"^i(?:'m| am) building\s+(.{2,120})$", "Current project"),
            (r"^(?:from now on,?\s+)?i prefer\s+(.{3,180})$", "Response preference"),
            (r"^i like my answers?\s+(.{3,180})$", "Response preference"),
            (r"^i don't like\s+(.{3,180})$", "Avoid preference"),
        ]
        for pattern, key in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip().rstrip(".!? ")
                if value and not _SENSITIVE.search(value):
                    candidates.append((key, value))
                break
        saved = []
        for key, value in candidates:
            self.set(key, value)
            saved.append({"key": key, "value": value})
        return saved

    def get_all(self):
        with self._connect() as db:
            rows = db.execute(
                "SELECT key,value,updated_at FROM memories ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, key):
        with self._connect() as db:
            cur = db.execute("DELETE FROM memories WHERE key=?", (key,))
            if cur.rowcount:
                db.execute("INSERT INTO memory_audit(key,action,timestamp) VALUES(?,?,?)",
                           (key, "deleted", datetime.now(timezone.utc).isoformat()))
        return cur.rowcount > 0

    def prompt_context(self, max_chars=1600):
        items = self.get_all()
        if not items:
            return ""
        lines = []
        used = 0
        for item in items[:40]:
            line = f"- {item['key']}: {item['value']}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
