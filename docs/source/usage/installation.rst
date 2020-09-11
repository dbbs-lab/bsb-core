=================
Istallation Guide
=================

The scaffold framework can be installed using Pip for Python 3

  .. code-block:: bash

    pip3 install bsb

You can verify that the installation works with

  .. code-block:: bash

    bsb -v=3 compile -x=200 -z=200 -p

This should generate an HDF5 file in your current directory and open a plot of
the generated network. If everything looks fine you are ready to advance to
the next topic.

Another verification method is to import the package in a Python script:

.. code-block:: python

  from bsb.core import Scaffold

  # Create a rather empty scaffold network with the default configuration.
  scaffold = Scaffold()

Simulator backends
==================

If you'd like to install the scaffold builder for point neuron simulations with NEST or multicompartmental neuron simulations with NEURON use:

.. code-block:: bash

  pip3 install bsb[nest]
  # or
  pip3 install bsb[neuron]
  # or both
  pip3 install bsb[nest,neuron]

.. note::

	This does not install the simulators, just the Python requirements for the framework
	to handle simulations using these backends.
