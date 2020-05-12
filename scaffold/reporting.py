import warnings, base64

warnings.filterwarnings("once", category=DeprecationWarning)

verbosity = 1
report_file = None


def set_verbosity(v):
    """
        Set the verbosity of the scaffold package.
    """
    global verbosity
    verbosity = v


def set_report_file(v):
    """
        Set a file to which the scaffold package should report instead of stdout.
    """
    global report_file
    report_file = v


preamble = chr(240) + chr(80) + chr(85) + chr(248) + chr(228)
preamble_bar = chr(191) * 3


def report(message, level=2, ongoing=False, token=None):
    """
        Send a message to the appropriate output channel.

        :param message: Text message to send.
        :type message: string
        :param level: Verbosity level of the message.
        :type level: int
        :param ongoing: The message is part of an ongoing progress report. This replaces the endline (`\\n`) character with a carriage return (`\\r`) character
    """
    if is_mpi_master and verbosity >= level:
        if report_file:
            with open(report_file, "a") as f:
                header = ""
                if token:
                    header += token
                header = base64.b64encode(bytes(header, "UTF-8")).decode("UTF-8")
                message = base64.b64encode(bytes(message, "UTF-8")).decode("UTF-8")
                f.write(preamble + header + preamble_bar + message + preamble)
        else:
            print(message, end="\n" if not ongoing else "\r")


def warn(message, category=None):
    """
        Send a warning.

        :param message: Warning message
        :type message: string
        :param category: The class of the warning.
    """
    if verbosity > 0:
        warnings.warn(message, category, stacklevel=2)


# Initialize MPI when this module is loaded, so that communications work even before
# any scaffold is created.

try:
    from mpi4py import MPI as _MPI

    try:
        # If neuron is installed, the user might want to use parallel NEURON
        # simulations. NEURON is incapable of properly initializing if MPI_Init
        # has already been called (which happens when you import MPI from mpi4py)
        # Therefore we must initialize NEURON first see
        # https://github.com/neuronsimulator/nrn/issues/428
        from patch import p

        # Initialize the ParallelContext singleton to properly initialize NEURON's
        # parallel simulation capabilities.
        _ = p.parallel
    except:
        pass

    MPI_rank = _MPI.COMM_WORLD.rank
    has_mpi_installed = True
    is_mpi_master = MPI_rank == 0
    is_mpi_slave = MPI_rank != 0
except ImportError:
    has_mpi_installed = False
    is_mpi_master = True
    is_mpi_slave = False
