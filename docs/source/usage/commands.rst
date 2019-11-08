#############################
List of command line commands
#############################

``scaffold [-v=1, -c=mouse_cerebellum] compile [-p, -o]``
---------------------------------------------------------

Compiles a network architecture: Places cells in a simulated volume and connects
them to eachother. All this information is then stored in a single HDF5 file.

.. include:: commands_defaults.txt

* ``-p``: Plot the created network.
* ``-o=<file>``, ``--output=<file>``: Output the result to a specific file.

``scaffold [-v=1, -c=mouse_cerebellum] simulate <name> --hdf5=<file>``
----------------------------------------------------------------------

Run a simulation from a compiled network architecture.

.. include:: commands_defaults.txt

* ``name``: Name of the simulation.
* ``--hdf5``: Path to the compiled network architecture.

``scaffold [-v=1, -c=mouse_cerebellum] run <name> [-p]``
--------------------------------------------------------

Run a simulation creating a new network architecture.

.. include:: commands_defaults.txt

* ``-p``: Plot the created network.
