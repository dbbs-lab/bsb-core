import os, sys, unittest, mpi4py, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from bsb.core import Scaffold, from_hdf5

# from scaffold.config import JSONConfig

scaffold_lookup = {}
mr_path = os.path.join(os.path.dirname(__file__), "..", "morphologies.h5")
mr_top_path = os.path.join(os.path.dirname(__file__), "..", "..", "morphologies.h5")
mr_rot_path = os.path.join(os.path.dirname(__file__), "..", "morpho_rotated.h5")
rotations_step = [30, 60]

_mpi_size = mpi4py.MPI.COMM_WORLD.Get_size()


def skip_parallel(o):
    return unittest.skipIf(_mpi_size > 1, "Skipped during parallel testing.")(o)


def single_process_test(o):
    import inspect

    if inspect.isclass(o) and issubclass(o, unittest.TestCase):
        return unittest.skipIf(_mpi_size > 1, "Single process test.")(o)
    elif callable(o):

        def wrapper(*args, **kwargs):
            if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
                o(*args, **kwargs)
            else:
                return

        return wrapper


def multi_process_test(o):
    import inspect

    if inspect.isclass(o) and issubclass(o, unittest.TestCase):
        return unittest.skipIf(_mpi_size < 2, "Multi process test.")(o)
    elif callable(o):

        def wrapper(*args, **kwargs):
            if _mpi_size > 1:
                o(*args, **kwargs)
            else:
                return

        return wrapper


def get_test_network(x=None, z=None):
    t = tuple([x, z])
    if not t in scaffold_lookup:
        _create_test_network(x, z)

    scaffold = from_hdf5(scaffold_lookup[t])
    return scaffold


def _create_test_network(*dimensions):
    scaffold_filename = "_test_network_{}_{}.hdf5".format(dimensions[0], dimensions[1])
    scaffold_lookup[tuple(dimensions)] = scaffold_filename
    scaffold_path = os.path.join(os.path.dirname(__file__), "..", scaffold_filename)
    if not os.path.exists(scaffold_path):
        # Fetch the default configuration file.
        config = JSONConfig(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "configs",
                "legacy_mouse_cerebellum.json",
            )
        )
        config.resize(*dimensions)
        print("Initializing {} by {} network for testing.".format(config.X, config.Z))
        # Resize the simulation volume to the specified dimensions
        # Set output filename
        config.output_formatter.file = scaffold_filename
        # Bootstrap the scaffold and create the network.
        scaffold = Scaffold(config)
        scaffold.compile_network()


def prep_morphologies():
    if not os.path.exists(mr_path):
        from bsb.output import MorphologyRepository as MR
        import dbbs_models

        mr = MR(mr_path)
        mr.import_arbz_module(dbbs_models)
        import shutil

        shutil.copyfile(mr_path, mr_top_path)


def prep_rotations():
    if not os.path.exists(mr_rot_path):
        from bsb.output import MorphologyRepository, MorphologyCache
        import dbbs_models

        mr = MorphologyRepository(mr_rot_path)
        mr.get_handle("w")
        mr.import_arbz("GranuleCell", dbbs_models.GranuleCell)
        mr.import_arbz("GolgiCell", dbbs_models.GolgiCell)
        mr.import_arbz("GolgiCell_A", dbbs_models.GolgiCell)
        mc = MorphologyCache(mr)
        mc.rotate_all_morphologies(rotations_step[0], rotations_step[1])


def get_config(file):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            file + ".json" if not file.endswith(".json") else "",
        )
    )


_exc_threads = {}


def excepthook(args, /):
    print("OLALALA IN HOOOK", file=sys.stderr)
    h = hash(args.thread)
    _exc_threads[h] = args.exc_value


threading.excepthook = excepthook


def timeout(timeout, abort=False):
    def decorator(f):
        def timed_f(*args, **kwargs):
            thread = threading.Thread(target=f, args=args, kwargs=kwargs)
            thread.start()
            thread.join(timeout=timeout)
            if thread.is_alive():
                if abort:
                    print(
                        TimeoutError(
                            1,
                            f"{f.__name__} timed out on rank {mpi4py.MPI.COMM_WORLD.Get_rank()}",
                            args,
                            kwargs,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    mpi4py.MPI.COMM_WORLD.Abort(1)
                raise TimeoutError(
                    1,
                    f"{f.__name__} timed out on rank {mpi4py.MPI.COMM_WORLD.Get_rank()}",
                    args,
                    kwargs,
                )
            elif hash(thread) in _exc_threads:
                e = _exc_threads[hash(thread)]
                del _exc_threads[hash(thread)]
                raise e

        return timed_f

    return decorator
