from functools import cache
import cProfile
import bsb.options
from bsb.services import MPI
from time import time
from uuid import uuid4
import pickle
import atexit
import sys
import traceback


class Meter:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self._starts = []
        self._stops = []

    def __enter__(self):
        self.start()

    def start(self):
        self._starts.append(time())

    def __exit__(self, exc, exc_type, tb):
        self.stop()
        self._exc = exc, exc_type, traceback.format_tb(tb)

    def stop(self):
        self._stops.append(time())

    @property
    def calls(self):
        return len(self._starts)

    @property
    def elapsed(self):
        return sum(e - s for s, e in zip(self._starts, self._stops))


class ProfilingSession:
    def __init__(self):
        self._started = False
        self._meters = []
        self.name = "bsb_profiling"
        self._current_f = None

    def set_name(self, name):
        self.name = name

    def start(self):
        if not self._started:
            self._started = True
            self.profile = cProfile.Profile()
            self.profile.enable()
            atexit.register(self.flush)

    def stop(self):
        if self._started:
            self._started = False
            self.profile.disable()
            atexit.unregister(self.flush)

    def meter(self, name, **data):
        meter = Meter(name, data)
        self._meters.append(meter)
        return meter

    def flush(self):
        profile = self.profile
        if self._current_f is None:
            uuid = uuid4()
            self._current_f = f"{self.name}_{MPI.get_rank()}_{uuid}"
        self.profile.dump_stats(f"{self._current_f}.prf")
        del self.profile
        with open(f"{self._current_f}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.profile = profile

    def view(self):
        try:
            from snakeviz.cli import main as snakeviz
        except ImportError:
            raise ImportError("Please `pip install snakeviz` to view profiles.") from None

        args = sys.argv
        if self._current_f is None:
            self.flush()
        sys.argv = ["snakeviz", f"{self._current_f}.prf"]

        snakeviz()

        sys.argv = args

    @staticmethod
    def load(fstem):
        with open(f"{fstem}.pkl", "rb") as f:
            return pickle.load(f)


@cache
def get_active_session():
    return ProfilingSession()


def activate_session(name=None):
    session = get_active_session()
    if name is not None:
        session.set_name(name)
    session.start()
    return session


def meter():
    def decorator(f):
        def decorated(*args, **kwargs):
            if bsb.options.profiling:
                session = get_active_session()
                with session.meter(f.__name__, args=str(args), kwargs=str(kwargs)):
                    r = f(*args, **kwargs)
                session.flush()
                return r
            else:
                return f(*args, **kwargs)

        return decorated

    return decorator


def view_profile(fstem):
    ProfilingSession.load(fstem).view()
