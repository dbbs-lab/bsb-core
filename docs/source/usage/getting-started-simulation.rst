##########################
Getting Started Simulation
##########################

=========================
NEST parallel simulations
=========================

To run parallel NEST simulations with the scaffold you can rely on NEST
parallelization (
https://nest-simulator.readthedocs.io/en/stable/guides/parallel_computing.html ).

Using MPI:

``mpirun -np 4 scaffold simulate your_simulation --hdf5=your_scaffold.hdf5``

Using OpenMP:

1. Set :guilabel:`threads` attribute of your simulation configuration to the
number of threads you want to use.

2. Run your simulation with

``scaffold simulate your_simulation --hdf5=your_scaffold.hdf5``
