"""Small explicit SQLite memory store for owner-approved preferences."""
import json
import sqlite3
from datetime import datetime, timezone

class MemoryStore:
    def __init__(self, path="orbit_state.sqlite3"):
        self.path=str(path)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("""CREATE TABLE IF NOT EXISTS memories(
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")
    def _connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.row_factory=sqlite3.Row
        return db
    def set(self,key,value):
        if not isinstance(key,str) or not key.strip() or len(key)>120:
            raise ValueError("Memory key must be 1-120 characters")
        if not isinstance(value,str) or len(value)>4000:
            raise ValueError("Memory value must be text up to 4000 characters")
        now=datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("INSERT INTO memories(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                       (key.strip(),value,now))
    def get_all(self):
        with self._connect() as db:
            rows=db.execute("SELECT key,value,updated_at FROM memories ORDER BY key").fetchall()
        return [dict(row) for row in rows]
    def delete(self,key):
        with self._connect() as db:
            cur=db.execute("DELETE FROM memories WHERE key=?",(key,))
        return cur.rowcount>0
    def prompt_context(self):
        items=self.get_all()
        if not items:return ""
        return "\n".join(f"- {x['key']}: {x['value']}" for x in items)
