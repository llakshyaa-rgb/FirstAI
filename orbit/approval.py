"""Explicit, per-phase owner approval gate for autonomous code changes."""
from __future__ import annotations
import threading
import uuid


class ApprovalGate:
    def __init__(self, store, timeout_seconds=86400):
        self.store = store
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._pending = {}

    def request(self, task_id, summary, files):
        approval_id = uuid.uuid4().hex
        event = threading.Event()
        record = {"event": event, "decision": None}
        with self._lock:
            self._pending[approval_id] = record
        self.store.emit(
            task_id, "approval_required",
            "Orbit is ready to modify its software. Explicit owner approval is required before these files are changed.",
            approval_id=approval_id, summary=str(summary), files=list(files),
            expires_in_seconds=self.timeout_seconds,
        )
        accepted = event.wait(self.timeout_seconds)
        with self._lock:
            self._pending.pop(approval_id, None)
        if not accepted:
            self.store.emit(task_id, "approval_expired", "No approval was received; no proposed changes were applied.")
            return False
        return record["decision"] is True

    def decide(self, approval_id, approved):
        if not isinstance(approval_id, str) or not approval_id:
            raise ValueError("approval_id is required")
        if type(approved) is not bool:
            raise ValueError("approved must be a boolean")
        with self._lock:
            record = self._pending.get(approval_id)
            if record is None:
                raise KeyError(approval_id)
            if record["event"].is_set():
                raise ValueError("This approval has already been decided")
            record["decision"] = approved
            record["event"].set()
        return {"approval_id": approval_id, "approved": approved}
