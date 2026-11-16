"""Run litmapr commands in the background for the Studio page.

Each job is the same command you would type in a terminal (`litmapr add ...`),
started as its own process. Its output is collected line by line so the page
can show a live log, and it can be stopped.

Jobs run one at a time, in the order they were asked for: most of them load a
model onto the GPU, and two at once would fight over its memory.
"""

import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

LITMAPR = [sys.executable, "-m", "litmapr"]
MAX_LINES = 5000  # a job's log keeps its last this-many lines
KEEP_FINISHED = 30  # finished jobs to remember


@dataclass
class Job:
    title: str
    args: list[str]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued, running, done, failed, stopped
    lines: list[str] = field(default_factory=list)
    dropped: int = 0  # lines cut from the start of a very long log
    code: int | None = None
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    stop_asked: bool = False

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed", "stopped")

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "command": "litmapr " + " ".join(_quote(a) for a in self.args),
            "status": self.status,
            "code": self.code,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "lines": self.dropped + len(self.lines),
        }

    def log_from(self, start: int) -> list[str]:
        """Lines from number `start` on (counting lines already dropped)."""
        return self.lines[max(0, start - self.dropped) :]


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or not arg) else arg


class JobRunner:
    """A queue of commands and one worker that runs them in order."""

    def __init__(self, program: list[str] | None = None, cwd: str | None = None):
        self.program = program or LITMAPR
        self.cwd = cwd or os.getcwd()
        self.jobs: dict[str, Job] = {}
        self.queue: deque[Job] = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.process: subprocess.Popen | None = None
        self.closed = False
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def submit(self, title: str, args: list[str]) -> Job:
        job = Job(title, args)
        with self.lock:
            self.jobs[job.id] = job
            self.queue.append(job)
            self._forget_old()
        self.wake.set()
        return job

    def stop(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.finished:
                return job
            job.stop_asked = True
            if job.status == "queued":
                self.queue.remove(job)
                job.status = "stopped"
                job.ended_at = time.time()
            elif self.process is not None:
                self.process.terminate()
        return job

    def all(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.queued_at, reverse=True)

    def busy(self) -> bool:
        with self.lock:
            return any(not j.finished for j in self.jobs.values())

    def close(self) -> None:
        """Stop everything (the server is shutting down)."""
        self.closed = True
        with self.lock:
            for job in self.queue:
                job.status, job.ended_at = "stopped", time.time()
            self.queue.clear()
            if self.process is not None:
                self.process.terminate()
        self.wake.set()

    def _forget_old(self) -> None:
        finished = sorted(
            (j for j in self.jobs.values() if j.finished), key=lambda j: j.queued_at
        )
        for job in finished[: max(0, len(finished) - KEEP_FINISHED)]:
            del self.jobs[job.id]

    def _work(self) -> None:
        while not self.closed:
            with self.lock:
                job = self.queue.popleft() if self.queue else None
            if job is None:
                self.wake.wait(timeout=1)
                self.wake.clear()
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env["COLUMNS"] = "100"  # keep Typer's help and error boxes readable
        try:
            process = subprocess.Popen(
                [*self.program, *job.args],
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            job.lines.append(f"Couldn't start the command: {error}")
            job.status, job.ended_at = "failed", time.time()
            return
        with self.lock:
            self.process = process
            job.status, job.started_at = "running", time.time()
        for line in process.stdout:
            job.lines.append(line.rstrip("\r\n"))
            if len(job.lines) > MAX_LINES:
                cut = len(job.lines) - MAX_LINES
                del job.lines[:cut]
                job.dropped += cut
        code = process.wait()
        with self.lock:
            self.process = None
            job.code = code
            job.ended_at = time.time()
            if job.stop_asked:
                job.status = "stopped"
                job.lines.append("Stopped.")
            else:
                job.status = "done" if code == 0 else "failed"
