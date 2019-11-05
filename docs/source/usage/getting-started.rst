###############
Getting Started
###############

===========
First steps
===========

The scaffold provides a simple command line interface (CLI) to compile network
architectures and run simulations.

Let's try out the most basic command, using the default configuration::

  scaffold -v=3 compile

This should produce prints and generate a timestamped HDF5 file in your current
directory.

You can explore the structure of the generated output by analysing it with the
scaffold shell. Open the scaffold shell like this::

  scaffold

You can now open and view the HDF5 file like this::

  open hdf5 <name>.hdf5
  view

This will print out the datasets and attributes in the output file. Most notably
this should give you access to the cell positions and connections.

See :doc:`/usage/cli` for a full guide.

============
First script
============
