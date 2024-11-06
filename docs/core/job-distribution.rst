BSB parallelization
===================

The BSB includes a full MPI interface to distribute the reconstruction jobs across the cores
allocated to the reconstruction and simulation of your networks.

Let us see how the jobs are created:

The BSB decomposes your `Scaffold` topology (`Region` and `Partition`) into a list of
:class:`Chunk <.storage._chunks.Chunk>` (parallelepipeds) of same size (and volume). The idea
is to consider the `Chunk` as independent from each other so that you can also decompose each
reconstruction task into a list of small tasks to apply to each `Chunk`. All reconstruction
jobs that happen after the creation of the topology follow this strategy.

The size and shape of the `Chunk` have a strong impact on the efficiency of the BSB parallelization.
Indeed, having a lot of very small `Chunk` would result in a lot of reading and writing in
the `Storage`. On the other hand, a short list of large `Chunk` will take longer to be completed
by each process, as bigger volumes imply a larger amount of constrains to fulfill.