from functools import cache
import cProfile
import bsb.options
from bsb.services import MPI


class ProfilingSession:
    def set_name(self, name):
        self.name = name

    def start(self):
        self.profile = cProfile.Profile()
        self.profile.enable()

    def stop(self):
        self.profile.disable()
        self.profile.dump_stats(f"profile_{self.name}_{MPI.get_rank()}.stats")


@cache
def get_active_session():
    return ProfilingSession()


def activate_session(name):
    session = get_active_session()
    session.set_name(name)
    return session


def is_session_active():
    return get_active_session.cache_info().hits > 0


def profile(f):
    def decorated(*args, **kwargs):
        if bsb.options.profiling and not is_session_active():
            session = activate_session(f.__name__)
            session.start()
            r = f(*args, **kwargs)
            session.stop()
            return r
        else:
            return f(*args, **kwargs)

    return decorated
