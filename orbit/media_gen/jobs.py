"""Threaded in-memory job queue with synchronized state snapshots."""
from __future__ import annotations

import copy
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from .base import MediaRequest, MediaResult


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    request: MediaRequest
    handler: Callable[[MediaRequest], MediaResult] = field(repr=False, compare=False)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    result: Optional[MediaResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobQueue:
    def __init__(self, workers: int = 2) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._workers = [
            threading.Thread(target=self._worker, daemon=True,
                             name=f"orbit-media-worker-{i + 1}")
            for i in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, request: MediaRequest,
               handler: Callable[[MediaRequest], MediaResult]) -> str:
        if not isinstance(request, MediaRequest):
            raise TypeError("request must be a MediaRequest")
        if not callable(handler):
            raise TypeError("handler must be callable")
        job = Job(request=request, handler=handler)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.copy(job) if job is not None else None

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            try:
                with self._lock:
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                result = job.handler(job.request)
                if not isinstance(result, MediaResult):
                    raise TypeError("job handler must return a MediaResult")
                with self._lock:
                    job.result = result
                    job.status = JobStatus.DONE if result.ok else JobStatus.FAILED
                    job.error = None if result.ok else (result.error or "handler returned failure")
            except Exception as exc:
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    job.finished_at = time.time()
                self._q.task_done()


global_queue = JobQueue(workers=2)
