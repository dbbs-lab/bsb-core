.. _faq:

###
FAQ
###

.. dropdown:: How to make the BSB work with NEST and MUSIC in an MPI context?
    :animate: fade-in-slide-down

    .. rubric:: Context

    When I simulate/reconstruct my BSB network in a context with `NEST` and `MUSIC` in parallel
    (with `MPI`), I encounter the following bug:

    .. code-block:: bash

        [ERROR] [2024.11.26 16:13:3 /path/to/nest-simulator/nestkernel/mpi_manager.cpp:134 @ MPIManager::init_mpi()] :
        When compiled with MUSIC, NEST must be initialized before any other modules that call MPI_Init(). Calling MPI_Abort().
        --------------------------------------------------------------------------
        MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
        with errorcode 1.

        NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
        You may or may not see output from other processes, depending on
        exactly when Open MPI kills them.
        --------------------------------------------------------------------------

    .. rubric:: Explanation

    This issue happens because MUSIC requires you to prepare the NEST context before the MPI context.
    In other words, you should import ``nest`` before importing ``mpi4py`` if you have installed NEST
    with MUSIC. Yet, the BSB leverages MPI for parallelizing the tasks of reconstructing or simulating
    your network, so it imports mpi4py.

    Using the BSB with NEST and MUSIC is therefore only possible through python scripts.
    At the start of the python scripts that needs to import bsb, add an extra line before to import
    nest even if you are not using it:

    .. code-block:: python

        import nest
        from bsb import Scaffold

        # rest of your code here.
