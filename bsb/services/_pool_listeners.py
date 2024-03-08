import abc
import datetime
from typing import cast

from .pool import PoolJobUpdateProgress, PoolProgress, PoolProgressReason


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
    def __call__(self, progress: PoolProgress):
        pass
