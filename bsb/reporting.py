import base64
import functools
import io
import sys
import warnings
from os import PathLike

from . import exceptions as _exc
from .services import MPI

_preamble = chr(240) + chr(80) + chr(85) + chr(248) + chr(228)
_preamble_bar = chr(191) * 3
_report_file = None


# Always show all scaffold warnings
for e in _exc.__dict__.values():
    if isinstance(e, type) and issubclass(e, Warning):
        warnings.simplefilter("always", e)


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)


def wrap_autoflush_stream(stream, writer):
    @functools.wraps(writer)
    def wrapped(self, *args, **kwargs):
        writer(*args, **kwargs)
        self.flush()

    return wrapped.__get__(stream)


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def in_pytest():
    return "pytest" in sys.modules


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


def read_report_file(file: PathLike):
    with open(file, "r") as f:
        print(f)
        return _decode(f.read())


def report(*message, level=2, ongoing=False, token=None, nodes=None, all_nodes=False):
    """
    Send a message to the appropriate output channel.

    :param message: Text message to send.
    :type message: str
    :param level: Verbosity level of the message.
    :type level: int
    :param ongoing: The message is part of an ongoing progress report.
    :type ongoing: bool
    """
    from . import options

    message = " ".join(map(str, message))
    rank = MPI.get_rank()
    if (
        (not rank and nodes is None) or all_nodes or (nodes is not None and rank in nodes)
    ) and options.verbosity >= level:
        if _report_file:
            with open(_report_file, "a") as f:
                f.write(_encode(token or "", message))
        else:
            print(message, end="\n" if not ongoing else "\r", flush=True)


def warn(message, category=None, stacklevel=2, log_exc=None):
    """
    Send a warning.

    :param message: Warning message
    :type message: str
    :param category: The class of the warning.
    """
    from . import options

    if log_exc:
        import traceback

        from .storage._util import cache

        log = f"{message}\n\n{traceback.format_exception(type(log_exc), log_exc, log_exc.__traceback__)}"
        id = cache.files.store(log)
        path = cache.files.id_to_file_path(id)
        message += f" See '{path}' for full error log."

    # Avoid infinite loop looking up verbosity when verbosity option is broken.
    if "Error retrieving option 'verbosity'" in message or options.verbosity > 0:
        if _report_file:
            with open(_report_file, "a") as f:
                f.write(_encode(str(category or "warning"), message))
        else:
            warnings.warn(message, category, stacklevel=stacklevel)


def _encode(header, message):
    header = base64.b64encode(bytes(header, "UTF-8")).decode("UTF-8")
    message = base64.b64encode(bytes(message, "UTF-8")).decode("UTF-8")
    return _preamble + header + _preamble_bar + message + _preamble


def _decode(payload: str):
    pos = -1
    plen = len(_preamble)
    reading = False
    log = []
    while (npos := payload.find(_preamble, pos + 1)) != -1:
        if reading:
            header, message = payload[pos + plen : npos].split(_preamble_bar)
            header = base64.b64decode(header).decode("UTF-8")
            message = base64.b64decode(message).decode("UTF-8")
            if header:
                message = f"[header: {header}] {message}"
            log.append(message)
        reading = not reading
        pos = npos
    return "\n".join(log)


def setup_reporting():
    warnings.formatwarning = warning_on_one_line
    # Don't touch stdout if we're in IPython
    if in_notebook() or in_pytest():
        return
    # Otherwise, tinker with stdout so that we autoflush after each write, better for MPI.
    try:
        stdout = open(sys.stdout.fileno(), "wb", 0)
        sys.stdout = io.TextIOWrapper(stdout, write_through=True)
    except io.UnsupportedOperation:  # pragma: nocover
        try:
            func_names = ["write", "writelines"]
            for func_name in func_names:
                method = getattr(sys.stdout, func_name)
                wrapped = wrap_autoflush_stream(sys.stdout, method)
                setattr(sys.stdout, func_name, wrapped)
        except Exception:
            warnings.warn(
                "Unable to create unbuffered wrapper around `sys.stdout`"
                + f" ({sys.stdout.__class__.__name__})."
            )
