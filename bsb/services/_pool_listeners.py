import abc
import contextlib
import datetime
from typing import cast

from blessed import Terminal
from dashing import HSplit, Log, Text

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
    def __init__(self):
        self._terminal = None

        self._ui = HSplit(
            Text(
                " ",
                title="Progress",
                color=7,
                border_color=1,
            ),
            Text(
                " ",
                title="Components",
                color=7,
                border_color=1,
            ),
            Log(title="Logs", color=7, border_color=1),
        )
        log = self._ui.items[-1]
        log.append("")

    @contextlib.contextmanager
    def _terminal_context(self, terminal: Terminal):
        with terminal.fullscreen(), terminal.cbreak(), terminal.hidden_cursor():
            yield

    def __enter__(self):
        self._terminal = Terminal()
        self._context = self._terminal_context(self._terminal)
        self._ui._terminal = self._terminal
        self._context.__enter__()
        self._ui.display()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, progress: PoolProgress):
        self._ui.display()
