Workflow parallelization
========================

The BSB includes a full `MPI` interface to distribute the reconstruction jobs across the cores
allocated to the reconstruction and simulation of your networks. You would have to install the
MPI library and the BSB interface to benefit from this feature (see the corresponding section in
the :doc:`/getting-started/installation`)


Job parallelization
-------------------

Let us see how the BSB jobs are created:

The BSB decomposes your `Scaffold` topology (`Region` and `Partition`) into a list of
:class:`Chunk <.storage._chunks.Chunk>` (parallelepipeds) of same size (and volume). The idea
is to consider the `Chunk` as independent from each other so that you can also decompose each
reconstruction task into a list of small tasks to apply to each `Chunk`. All reconstruction
jobs that happen after the creation of the topology follow this strategy.

The size and shape of the `Chunk` have a strong impact on the efficiency of the BSB parallelization.
Indeed, having a lot of very small `Chunk` would result in a lot of reading and writing in
the `Storage`. On the other hand, a short list of large `Chunk` will take longer to be completed
by each process, as bigger volumes imply a larger amount of constrains to fulfill.

The Chunks' dimensions in µm can be set in the `Configuration` in the ``Network`` node:

.. tab-set-code::

    .. code-block:: json

        "network": {
          "x": 200.0,
          "y": 200.0,
          "z": 200.0,
          "chunk_size": [50, 20, 10]
        }

    .. code-block:: yaml

        network:
          x: 200
          y: 200
          z: 200
          chunk_size:
            - 50
            - 20
            - 10

    .. code-block:: python

        config.network.x = 200.0
        config.network.y = 200.0
        config.network.z = 200.0
        config.network.chunk_size = [50, 20, 10]

As you can see the dimensions of the Chunks can differ according to each dimension.

Running the BSB in parallel
---------------------------

You can run any bsb command or python scripts with MPI with the ``mpirun`` command,
specifying the number of core to allocate:

.. code-block:: bash

    # run the BSB reconstruction with 5 cores
    mpirun -n 5 bsb compile my-config.json -v 3

    # run a python script in parallel with 4 cores
    mpirun -n 4 python my-script.py
