import warnings, base64, io, sys, functools


def wrap_writer(stream, writer):
    @functools.wraps(writer)
    def wrapped(self, *args, **kwargs):
        writer(*args, **kwargs)
        self.flush()

    return wrapped.__get__(stream)


try:
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)
except io.UnsupportedOperation:  # pragma: nocover
    try:
        writers = ["write", "writelines"]
        for w in writers:
            writer = getattr(sys.stdout, w)
            wrapped = wrap_writer(sys.stdout, writer)
            setattr(sys.stdout, w, wrapped)
    except:
        warnings.warn(
            f"Unable to create unbuffered wrapper around `sys.stdout` ({sys.stdout.__class__.__name__})."
        )

_verbosity = 1
_report_file = None


def set_verbosity(v):
    """
    Set the verbosity of the scaffold package.
    """
    global _verbosity
    _verbosity = v


def get_verbosity():
    """
    Return the verbosity of the scaffold package.
    """
    return _verbosity


def set_report_file(v):
    """
    Set a file to which the scaffold package should report instead of stdout.
    """
    global _report_file
    _report_file = v


def get_report_file():
    """
    Return the report file of the scaffold package.
    """
    return _report_file


preamble = chr(240) + chr(80) + chr(85) + chr(248) + chr(228)
preamble_bar = chr(191) * 3


def report(*message, level=2, ongoing=False, token=None, nodes=None, all_nodes=False):
    """
    Send a message to the appropriate output channel.

    :param message: Text message to send.
    :type message: string
    :param level: Verbosity level of the message.
    :type level: int
    :param ongoing: The message is part of an ongoing progress report. This replaces the endline (`\\n`) character with a carriage return (`\\r`) character
    """
    message = " ".join(map(str, message))
    if (
        (is_mpi_master and nodes is None)
        or all_nodes
        or (nodes is not None and MPI_rank in nodes)
    ) and _verbosity >= level:
        if _report_file:
            with open(_report_file, "a") as f:
                f.write(_encode(token or "", message))
        else:
            print(message, end="\n" if not ongoing else "\r", flush=True)


def warn(message, category=None):
    """
    Send a warning.

    :param message: Warning message
    :type message: str
    :param category: The class of the warning.
    """
    if _verbosity > 0:
        if _report_file:
            with open(_report_file, "a") as f:
                f.write(_encode(str(category or "warning"), message))
        else:
            warnings.warn(message, category, stacklevel=2)


def _encode(header, message):
    header = base64.b64encode(bytes(header, "UTF-8")).decode("UTF-8")
    message = base64.b64encode(bytes(message, "UTF-8")).decode("UTF-8")
    return preamble + header + preamble_bar + message + preamble


# Initialize MPI when this module is loaded, so that communications work even before
# any scaffold is created.

try:
    try:
        import nest
    except:
        pass
    from mpi4py import MPI as _MPI

    MPI_rank = _MPI.COMM_WORLD.rank
    has_mpi_installed = True
    is_mpi_master = MPI_rank == 0
    is_mpi_slave = MPI_rank != 0
except ImportError:
    has_mpi_installed = False
    is_mpi_master = True
    is_mpi_slave = False

report("Reporting module initialised.", level=4)
