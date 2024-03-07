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

.. note::

  Depends on the MPI service.

JobPool
=======

The ``JobPool`` service allows you to ``submit`` ``Jobs`` and then ``execute`` them.

.. note::

  Depends on the MPI service.

Most component types have a ``queue`` method that takes a job pool as an argument and
lets them schedule their jobs.

The recommended way to open a job pool is to use :meth:`~bsb.core.Scaffold.create_job_pool`:

.. code-block:: python

  network = from_storage("example.hdf5")
  pool = network.create_job_pool()
  if pool.is_main():
    # Only the main node needs to schedule the jobs
    for component in network.placement.values():
      component.queue(pool)
  # But everyone needs to partake in the execute call
  pool.execute()

On top of opening the job pool this also registers the appropriate listeners. Listeners
listen to updates emitted by the job pool and can respond to changes, for example by printing
them out to display the progress of the job pool:

.. code-block:: python

  _t = None
  def report_time_elapsed(progress):
    global _t
    if progress.reason == PoolProgressReason.POOL_STATUS_CHANGE:
      if progress.status == PoolStatus.STARTING:
        _t = time.time()
      elif progress.status == PoolStatus.ENDING:
        print(f"Pool execution finished. {time.time()} seconds elapsed.")

  pool = network.create_job_pool()
  pool.add_listener(report_time_elapsed)
  pool.submit(lambda scaffold: time.sleep(2))
  pool.execute()
  # Will print `Pool execution finished. 2 seconds elapsed.`