"""In-memory job queue with worker threads. Swap for Celery/RQ in prod."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .base import MediaRequest, MediaResult


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    request: MediaRequest
    handler: Callable[[MediaRequest], MediaResult]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    result: Optional[MediaResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobQueue:
    def __init__(self, workers: int = 2) -> None:
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._workers = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(workers)
        ]
        for t in self._workers:
            t.start()

    def submit(self, request: MediaRequest, handler: Callable[[MediaRequest], MediaResult]) -> str:
        job = Job(request=request, handler=handler)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            try:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                job.result = job.handler(job.request)
                job.status = JobStatus.DONE if job.result.ok else JobStatus.FAILED
                if not job.result.ok:
                    job.error = job.result.error
            except Exception as e:  # noqa: BLE001
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                job.finished_at = time.time()
                self._q.task_done()


global_queue = JobQueue(workers=2)
