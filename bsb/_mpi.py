try:
    from mpi4py import MPI as _MPI

    MPI_rank = _MPI.COMM_WORLD.rank
    has_mpi_installed = True
    is_mpi_master = MPI_rank == 0
    is_mpi_slave = MPI_rank != 0
except ImportError:
    has_mpi_installed = False
    is_mpi_master = True
    is_mpi_slave = False
