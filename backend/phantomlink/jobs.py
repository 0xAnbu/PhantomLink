"""Single-slot background job runner (one training at a time) with an append-only event log."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class JobBusy(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    kind: str
    config: dict
    status: str = "running"            # running | completed | stopped | failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    result: dict | None = None
    events: list[dict] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event)

    def public(self, with_events: bool = False) -> dict:
        d = {"id": self.id, "kind": self.kind, "config": self.config, "status": self.status,
             "started": self.started, "finished": self.finished, "error": self.error,
             "result": self.result, "n_events": len(self.events)}
        if with_events:
            d["events"] = self.events
        return d


class JobManager:
    MAX_EVENTS = 5000

    def __init__(self):
        self._lock = threading.Lock()
        self.current: Job | None = None
        self.jobs: dict[str, Job] = {}

    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def start(self, kind: str, config: dict,
              fn: Callable[[Callable[[dict], None], Callable[[], bool]], dict | None]) -> Job:
        with self._lock:
            if self.busy():
                raise JobBusy(f"ya hay un entrenamiento en curso ({self.current.kind})")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, config=config)
            self.current = job
            self.jobs[job.id] = job

        def emit(event: dict) -> None:
            event = {"seq": len(job.events), "t": time.time(), **event}
            if len(job.events) < self.MAX_EVENTS or event.get("type") != "metrics":
                job.events.append(event)

        def runner() -> None:
            try:
                job.result = fn(emit, job.stop_event.is_set)
                job.status = "stopped" if job.stop_event.is_set() else "completed"
            except Exception as exc:  # surfaced to the UI, never swallowed
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                emit({"type": "error", "message": job.error, "trace": traceback.format_exc()[-2000:]})
            finally:
                job.finished = time.time()
                emit({"type": "end", "status": job.status})

        threading.Thread(target=runner, name=f"job-{job.id}", daemon=True).start()
        return job

    def stop(self) -> Job | None:
        job = self.current
        if job and job.status == "running":
            job.stop_event.set()
            return job
        return None

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def wait(self, job_id: str, timeout: float = 600) -> Job:
        """Block until a job ends (used by tests and scripts)."""
        job = self.jobs[job_id]
        t0 = time.time()
        while job.status == "running" and time.time() - t0 < timeout:
            time.sleep(0.05)
        return job


def jsonable(obj: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
