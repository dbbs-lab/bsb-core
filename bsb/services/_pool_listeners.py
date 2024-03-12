import abc
import contextlib
import datetime
import time
from collections import defaultdict
from typing import cast

from blessed import Terminal
from dashing import HSplit, Log, Text

from .pool import (
    Job,
    JobStatus,
    PoolJobUpdateProgress,
    PoolProgress,
    PoolProgressReason,
    PoolStatus,
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
    def __init__(self):
        self._terminal = None
        self._last_update = 0

        self._ui = HSplit(
            Text(
                " ",
                title="Progress",
                color=7,
                border_color=3,
            ),
            Text(
                " ",
                title="Components",
                color=7,
                border_color=3,
            ),
            Log(title="Logs", color=7, border_color=3),
        )

    @contextlib.contextmanager
    def _terminal_context(self, terminal: Terminal):
        with terminal.fullscreen(), terminal.cbreak(), terminal.hidden_cursor():
            yield

    def __enter__(self):
        self._terminal = Terminal()
        self._context = self._terminal_context(self._terminal)
        self._ui._terminal = self._terminal
        self._context.__enter__()
        self.update()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, progress: PoolProgress):
        if (
            progress.reason == PoolProgressReason.POOL_STATUS_CHANGE
            and progress.status == PoolStatus.EXECUTING
        ):
            self._tally = PoolTally(JobTally)
            for job in progress.jobs:
                self._tally[job.name].tally(job)
            cast(Text, self._ui.items[1]).text = "\n".join(
                f"{k}: {v}" for k, v in self._tally.sorted()
            )
        if (
            (progress.reason == PoolProgressReason.JOB_STATUS_CHANGE)
            and progress.status != JobStatus.PENDING
            and progress.status != JobStatus.QUEUED
        ):
            self._tally.update_tally(cast(PoolJobUpdateProgress, progress))
            cast(Text, self._ui.items[1]).text = (
                "\n".join(f"{k}: {v}" for k, v in self._tally.top()) or " "
            )
        self.update()

    def update(self):
        if time.time() - self._last_update >= 1 / 25:
            self._ui.display()
            self._last_update = time.time()


class JobTally:
    def __init__(self):
        self._total = 0
        self._tallies = defaultdict(int)

    def tally(self, job: Job):
        self._tallies[job.status] += 1
        self._total += 1

    def tally_unfinished(self):
        return (
            self._tallies[JobStatus.PENDING]
            + self._tallies[JobStatus.QUEUED]
            + self._tallies[JobStatus.RUNNING]
        )

    def tally_finished(self):
        return self._total - self.tally_unfinished()

    def progress(self):
        return self.tally_finished() / self._total

    def finished(self):
        return self.tally_unfinished() == 0

    def update(self, old_status: JobStatus, new_status: JobStatus):
        self._tallies[old_status] = max(0, self._tallies[old_status] - 1)
        self._tallies[new_status] += 1

    def __str__(self):
        return f"({self.tally_finished()}/{self._total})"


class PoolTally(defaultdict):
    def sorted(self):
        yield from sorted(self.items(), key=lambda i: -i[1].progress())

    def top(self):
        yield from (i for i in self.sorted() if not i[1].finished())

    def update_tally(self, progress: PoolJobUpdateProgress):
        self[progress.job.name].update(progress.old_status, progress.status)
