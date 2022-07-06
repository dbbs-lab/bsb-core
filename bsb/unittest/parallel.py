import unittest as _unittest
import inspect as _inspect
import mpi4py as _mpi4py
import threading as _threading

MPI = _mpi4py.MPI.COMM_WORLD
_mpi_size = MPI.Get_size()


def skip_parallel(o):
    return _unittest.skipIf(_mpi_size > 1, "Skipped during parallel testing.")(o)


def single_process_test(o):
    if _inspect.isclass(o) and issubclass(o, _unittest.TestCase):
        return _unittest.skipIf(_mpi_size > 1, "Single process test.")(o)
    elif callable(o):

        def wrapper(*args, **kwargs):
            if MPI.Get_rank() == 0:
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
        def guarded_f(*args, **kwargs):
            response = f"{f.__name__}____start"
            buff = MPI.allgather(response)
            if any(b != response for b in buff):
                raise TimeoutError(f"Start token desynchronization: {buff}")
            f(*args, **kwargs)
            response = f"{f.__name__}____end"
            buff = MPI.allgather(response)
            if any(b != response for b in buff):
                raise TimeoutError(f"End token desynchronization: {buff}")

        def timed_f(*args, **kwargs):
            thread = _threading.Thread(target=guarded_f, args=args, kwargs=kwargs)
            thread.start()
            thread.join(timeout=timeout)
            if thread.is_alive():
                err = TimeoutError(
                    1,
                    f"{f.__name__} timed out on rank {MPI.Get_rank()}",
                    args,
                    kwargs,
                )
                if abort:
                    import traceback

                    errlines = traceback.format_exception(
                        type(err), err, err.__traceback__
                    )
                    print(
                        *errlines,
                        file=sys.stderr,
                        flush=True,
                    )
                    MPI.Abort(1)
                raise err
            elif hash(thread) in _exc_threads:
                e = _exc_threads[hash(thread)]
                del _exc_threads[hash(thread)]
                raise e

        return timed_f

    return decorator
