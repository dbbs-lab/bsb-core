######################
Command Line Interface
######################

There are 2 entry points in the command line interface:

* **A command**: Can be written in a command line prompt such as the Terminal on
  Linux or CMD on Windows.
* **The shell**: Can be opened by giving typing ``scaffold`` into a command line
  prompt.

Take a look at the :doc:`commands`.

**************
Scaffold shell
**************

The scaffold shell is an interactive environment where commands can be given.
Unlike with the command line your state is maintained in between commands.

Opening the shell
=================

Open your favorite command line prompt and if the scaffold package is succesfully
installed the ``scaffold`` command should be available.

You can close the shell by typing ``exit``.

The base state
==============

After opening the shell it will be in the base (default) state. In this state
you have access to several commands like opening morphology repositories or hdf5
files.

List of base commands
---------------------

* ``open mr <filename>``: Open a morphology repository. See :ref:`repl_mr_commands`
* ``open hdf5 <filename>``: Open an HDF5 file. See :ref:`repl_hdf5_commands`

The morphology repository state
===============================

In this state you can modify the morphology repository.
After you've opened a repository the shell will display a prefix::

  repo <filename>:

.. _repl_mr_commands:

List of mr commands
-------------------

* ``list all``: Show a list of all morphologies available in the repository.
* ``list voxelized``: Show a list of all morphologies with voxel cloud
  information available.
* ``import repo <filename>``: Import all morphologies from another repository
  * ``-f``/``--overwrite``: Overwrite existing morphologies.
* ``import swc <file> <name>``: Import an SWC morphology and store it under the
  given name.
* ``remove <name>``: Remove a morphology from the repository.
* ``voxelize <name> [<n=130>]``: Generate a voxel cloud of ``n`` (optional,
  default=130) voxels for the morphology.
* ``plot <name>``: Plot the morphology.
* ``close``: Exit the mr state.

The HDF5 state
==============

In this state you can view the structure of HDF5 files.

.. _repl_hdf5_commands:

List of hdf5 commands:
----------------------

* ``view``: Create a hierarchical print of the HDF5 file, groups, datasets, and
  attributes.
