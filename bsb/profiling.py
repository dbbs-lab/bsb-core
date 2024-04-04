import atexit
import cProfile
import functools
import pickle
import sys
import traceback
import warnings
from functools import cache
from time import time
from uuid import uuid4

from .services import MPI


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
        self._flushcounter = 0

    def set_name(self, name):
        self.name = name

    @property
    def meters(self):
        return self._meters.copy()

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

    def flush(self, stats=True):
        profile = self.profile
        if self._current_f is None:
            uuid = uuid4()
            self._current_f = f"{self.name}_{MPI.get_rank()}_{uuid}"
        if stats:
            self.profile.dump_stats(f"{self._current_f}_{self._flushcounter}.prf")
            self._flushcounter += 1
        try:
            del self.profile
            with open(f"{self._current_f}.pkl", "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            warnings.warn(f"Could not store profile: {e}")
        finally:
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


def node_meter(*methods):
    def get_node_method_name(method, args, kwargs):
        return (
            f"{args[0].get_node_name()}[{args[0].__class__.__name__}].{method.__name__}"
        )

    def decorator(node_cls):
        for method_name in methods:
            if method := getattr(node_cls, method_name, None):
                setattr(node_cls, method_name, meter(method, name_f=get_node_method_name))

        return node_cls

    return decorator


def meter(f=None, *, name_f=None):
    def decorated(*args, **kwargs):
        import bsb.options

        if bsb.options.profiling:
            session = get_active_session()
            if name_f:
                name = name_f(f, args, kwargs)
            else:
                name = f.__name__
            with session.meter(name, args=str(args), kwargs=str(kwargs)):
                r = f(*args, **kwargs)
            session.flush(stats=False)
            return r
        else:
            return f(*args, **kwargs)

    if f is None:

        def decorator(g):
            nonlocal f
            f = g
            return functools.wraps(f)(decorated)

        return decorator
    else:
        return functools.wraps(f)(decorated)


def view_profile(fstem):
    ProfilingSession.load(fstem).view()


__all__ = [
    "Meter",
    "ProfilingSession",
    "activate_session",
    "get_active_session",
    "meter",
    "node_meter",
    "view_profile",
]
