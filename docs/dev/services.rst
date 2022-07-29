Services
########

The BSB provides some "services", which can be provided by a fallback system of providers.
Usually they import a package, and if it isn't found, provide a sensible mock, or an
object that errors on first use, so that the framework and any downstream packages can
always import and use the service (if a mock is provided).

MPI
===

The MPI service provided by :attr:`bsb.services.MPI` is the ``COMM_WORLD``
:class:`mpi4py.MPI.Comm` if ``mpi4py`` is available, otherwise it is an emulator that
emulates a single node parallel context.

.. error::

  If any environment variables are present that contain ``MPI`` in their name an error is
  raised, as execution in an actual MPI environment won't work without ``mpi4py``.

MPILock
=======

The MPILock service provides ``mpilock``'s ``WindowController`` if it is available, or a
mock that immediately and unconditionally acquires its lock and continues.

.. error::

	Depends on the MPI service. Will error out under MPI conditions.

JobPool
=======

The ``JobPool`` service allows you to ``submit`` ``Jobs`` and then ``execute`` them.

.. error::

	Depends on the MPI service. Will error out under MPI conditions.
