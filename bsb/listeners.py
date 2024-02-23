import abc
import datetime
import sys
import traceback
from typing import cast

from .services.pool import (
    Job,
    JobStatus,
    PoolJobUpdateProgress,
    PoolProgress,
    PoolProgressReason,
    PoolStatus,
)


class Listener(abc.ABC):
    def __init__(self, fail_fast=True):
        self._ff = fail_fast

    @abc.abstractmethod
    def __call__(self, progress: PoolProgress):
        pass


class NonTTYTerminalListener(Listener):
    def __call__(self, progress: PoolProgress):
        if progress.reason == PoolProgressReason.JOB_STATUS_CHANGE:
            job = cast(PoolJobUpdateProgress, progress).job
            print(f"[{datetime.datetime.now()}] {job.status} {job.name}")
            if self._ff and job.error:
                raise job.error
        if progress.reason == PoolProgressReason.MAX_TIMEOUT_PING:
            print(f"[{datetime.datetime.now()}] Progress ping.")


class TTYTerminalListener(Listener):
    def __call__(self, jobs: list[Job], status: PoolStatus):
        if status == PoolStatus.STARTING:
            # todo: initialize fancy curses window
            pass
        elif status == PoolStatus.RUNNING:
            # todo: update window
            pass
        elif status == PoolStatus.ENDING:
            # todo: stop fancy curses window
            pass
