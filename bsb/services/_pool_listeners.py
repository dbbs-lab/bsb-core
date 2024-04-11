import abc
import contextlib
import datetime
import time
from collections import defaultdict
from typing import cast

from blessed import Terminal
from dashing import Color, DoubleColumn, HSplit, Text

from . import MPI
from .pool import (
    Job,
    JobStatus,
    PoolJobAddedProgress,
    PoolJobUpdateProgress,
    PoolProgress,
    PoolProgressReason,
    PoolStatus,
    Workflow,
)


class Listener(abc.ABC):
    @abc.abstractmethod
    def __call__(self, progress: PoolProgress):
        pass


class NonTTYTerminalListener(Listener):
    def __call__(self, progress: PoolProgress):
        if progress.reason == PoolProgressReason.JOB_STATUS_CHANGE:
            job = cast(PoolJobUpdateProgress, progress).job
            print(f"[{datetime.datetime.now()} - BSB] {job.status} {job.name}")
        if progress.reason == PoolProgressReason.MAX_TIMEOUT_PING:
            print(f"[{datetime.datetime.now()}] Progress ping.")


class TTYTerminalListener(Listener):
    def __init__(self, fps: int):
        self._workflow: Workflow = None
        self._terminal = None
        self._last_update = 0
        self.fps = fps

        self._ui = HSplit(
            Text(
                "",
                title="Progress",
                color=Color.White,
                border_color=Color.Yellow,
            ),
            DoubleColumn(
                [],
                title="Jobs",
                color=7,
                border_color=3,
            ),
        )

    @contextlib.contextmanager
    def _terminal_context(self, terminal: Terminal):
        with terminal.fullscreen(), terminal.cbreak(), terminal.hidden_cursor():
            yield

    def __enter__(self):
        if not MPI.get_rank():
            self._terminal = Terminal()
            self._context = self._terminal_context(self._terminal)
            self._ui._terminal = self._terminal
            self._context.__enter__()
            self._start = time.perf_counter()
            self.update()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MPI.get_rank():
            self._context.__exit__(exc_type, exc_val, exc_tb)

    def _initialize(self, progress: PoolProgress):
        self._tally = PoolTally(JobTally)
        if progress.workflow:
            self._workflow = progress.workflow

    def __call__(self, progress: PoolProgress):
        if MPI.get_rank():
            return
        if progress.reason == PoolProgressReason.POOL_STATUS_CHANGE:
            if progress.status == PoolStatus.SCHEDULING:
                self._initialize(progress)
            self._pool_status = progress.status
        if progress.reason == PoolProgressReason.JOB_ADDED:
            job = cast(PoolJobAddedProgress, progress).job
            self._tally[job.name].tally(job)
            self.update_jobs()
        self.update_progress()
        if (
            (progress.reason == PoolProgressReason.JOB_STATUS_CHANGE)
            and progress.status != JobStatus.PENDING
            and progress.status != JobStatus.QUEUED
        ):
            self._tally.update_tally(cast(PoolJobUpdateProgress, progress))
            self.update_jobs()
        self.update()

    def update(self):
        if time.time() - self._last_update >= 1 / self.fps:
            self._ui.display()
            self._last_update = time.time()

    def update_jobs(self):
        cast(DoubleColumn, self._ui.items[1]).lines = [
            (name, str(tally)) for name, tally in self._tally.top()
        ]

    def update_progress(self):
        cast(Text, self._ui.items[0]).text = (
            (f"Workflow: {' | '.join(self._workflow.phases)}\n" if self._workflow else "")
            + (f"Phase: {self._workflow.phase}\n" if self._workflow else "")
            + f"Status: {str(self._pool_status).split('.')[1]}\n"
            f"Job total:\n"
            f" {self._tally}\n"
            f"Elapsed: {round(time.perf_counter() - self._start, 2)}s\n"
            f"Job est.: {round(self._tally.get_estimate(), 2)}s"
        )


class JobTally:
    def __init__(self):
        self._total = 0
        self._tallies = defaultdict(int)

    @property
    def total(self):
        return self._total

    def tally(self, job: Job):
        self._tallies[job.status] += 1
        self._total += 1

    def tally_unfinished(self):
        return self.tally_status(JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)

    def tally_finished(self):
        return self._total - self.tally_unfinished()

    def tally_status(self, *status: JobStatus):
        return sum((self._tallies[s] for s in status), start=0)

    def progress(self):
        return self.tally_finished() / self._total

    def finished(self):
        return self.tally_unfinished() == 0

    def update(self, old_status: JobStatus, new_status: JobStatus):
        self._tallies[old_status] = max(0, self._tallies[old_status] - 1)
        self._tallies[new_status] += 1

    def __str__(self):
        others = ""
        if running := self.tally_status(JobStatus.RUNNING):
            others += f"→ {running}/"
        if failed := self.tally_status(JobStatus.FAILED, JobStatus.ABORTED):
            others += f"🗙 {failed}/"
        if skipped := self.tally_status(JobStatus.CANCELLED):
            others += f"⊘ {skipped}/"
        return f"({others}✔ {self.tally_status(JobStatus.SUCCESS)}/{self._total})"


class PoolTally(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elapsed = {}
        self._durations = defaultdict(lambda: [0, 0])

    def sorted(self):
        yield from sorted(self.items(), key=lambda i: -i[1].progress())

    def top(self):
        yield from (i for i in self.sorted() if not i[1].finished())

    def update_tally(self, progress: PoolJobUpdateProgress):
        self[progress.job.name].update(progress.old_status, progress.status)
        if progress.status == JobStatus.RUNNING:
            self._elapsed[id(progress.job)] = time.perf_counter()
        elif progress.old_status == JobStatus.RUNNING:
            total, n = self._durations[progress.job.name]
            self._durations[progress.job.name] = [
                total + time.perf_counter() - self._elapsed[id(progress.job)],
                n + 1,
            ]
            del self._elapsed[id(progress.job)]

    def tally(self, *status: JobStatus):
        if status:
            return sum(tally.tally_status(*status) for tally in self.values())
        else:
            return sum(tally.total for tally in self.values())

    def get_estimate(self):
        return sum(
            total / n * self[job_name].tally_unfinished()
            for job_name, (total, n) in self._durations.items()
            if n
        )

    def __str__(self):
        others = ""
        if running := self.tally(JobStatus.RUNNING):
            others += f"→ {running}/"
        if failed := self.tally(JobStatus.FAILED, JobStatus.ABORTED):
            others += f"🗙 {failed}/"
        if skipped := self.tally(JobStatus.CANCELLED):
            others += f"⊘ {skipped}/"
        return f"({others}✔ {self.tally(JobStatus.SUCCESS)}/{self.tally()})"
