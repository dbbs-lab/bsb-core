import unittest as _unittest
import inspect as _inspect
import threading as _threading
import http.client as _http
from bsb.services import MPI


_mpi_size = MPI.get_size()


def internet_connection():
    for ip in ("1.1.1.1", "8.8.8.8"):
        conn = _http.HTTPSConnection("8.8.8.8", timeout=2)
        try:
            _http.request("HEAD", "/")
            return True
        except Exception:
            pass
        finally:
            conn.close()
    else:
        return False


def skip_nointernet(o):
    return _unittest.skipIf(not internet_connection(), "Internet connection required.")(o)


def skip_parallel(o):
    return _unittest.skipIf(_mpi_size > 1, "Skipped during parallel testing.")(o)


def single_process_test(o):
    if _inspect.isclass(o) and issubclass(o, _unittest.TestCase):
        return _unittest.skipIf(_mpi_size > 1, "Single process test.")(o)
    elif callable(o):

        def wrapper(*args, **kwargs):
            if MPI.get_rank() == 0:
                o(*args, **kwargs)
            else:
                return

        return wrapper


def multi_process_test(o):
    if _inspect.isclass(o) and issubclass(o, _unittest.TestCase):
        return _unittest.skipIf(_mpi_size < 2, "Multi process test.")(o)
    elif callable(o):

        def wrapper(*args, **kwargs):
            if _mpi_size > 1:
                o(*args, **kwargs)
            else:
                return

        return wrapper


_exc_threads = {}


def _excepthook(args, /):
    h = hash(args.thread)
    _exc_threads[h] = args.exc_value


_threading.excepthook = _excepthook


def timeout(timeout, abort=False):
    def decorator(f):
        def timed_f(*args, **kwargs):
            thread = _threading.Thread(target=f, args=args, kwargs=kwargs)
            thread.start()
            thread.join(timeout=timeout)
            try:
                if thread.is_alive():
                    err = TimeoutError(
                        1,
                        f"{f.__name__} timed out on rank {MPI.get_rank()}",
                        args,
                        kwargs,
                    )
                    raise err
                elif hash(thread) in _exc_threads:
                    e = _exc_threads[hash(thread)]
                    del _exc_threads[hash(thread)]
                    raise e
            except Exception as e:
                import traceback, sys

                errlines = traceback.format_exception(type(e), e, e.__traceback__)
                print(
                    *errlines,
                    file=sys.stderr,
                    flush=True,
                )
                if MPI.get_size() > 1:
                    MPI.abort(1)

        return timed_f

    return decorator


def on_main_only(f):
    def main_wrapper(*args, **kwargs):
        if MPI.get_rank():
            MPI.barrier()
        else:
            r = f(*args, **kwargs)
            MPI.barrier()
            return r

    return main_wrapper


def serial_setup(cls):
    cls.setUp = on_main_only(cls.setUp)
    return cls
