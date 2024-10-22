.. _installation-guide:

============
Installation
============
| The BSB framework is tested and documented for Python versions 3.9, 3.10 and 3.11.
| Support and compatibility are not guaranteed for the other versions of Python.

.. tip::

  We highly recommend you to use
  `python virtual environments <https://realpython.com/python-virtual-environments-a-primer/>`_
  to install BSB!


The BSB framework can be installed using ``pip``:

.. code-block:: bash

  pip install bsb

You can verify that the installation works with:

.. code-block:: python

  from bsb import Scaffold

  # Create an empty scaffold network with the default configuration.
  scaffold = Scaffold()

You have now the minimal installation required to complete the :doc:`getting started <top-level-guide>` section.

Parallel support
================

The BSB parallelizes the network reconstruction using MPI, and translates simulator
instructions to the simulator backends with it as well, for effortless parallel
simulation. To use MPI from Python the `mpi4py
<https://mpi4py.readthedocs.io/en/stable/>`_ package is required, which in turn needs a
working MPI implementation installed in your environment.

On your local machine, first install MPI:

.. tab-set::

  .. tab-item:: Ubuntu
    :sync: bash

      .. code-block:: bash

        sudo apt-get update && sudo apt-get install -y libopenmpi-dev openmpi-bin

  .. tab-item:: MacOS
    :sync: bash

      .. code-block:: bash

        # You need to have xcode installed.
        # For Homebrew
        brew install openmpi

        # For MacPorts
        port install openmpi


  .. tab-item:: Windows

    For Windows, you would need to install C/C++ compiler as well as the
    `Microsoft MPI <https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi>`_
    program.

On supercomputers it is usually installed already, otherwise contact your administrator.

To then install the BSB with parallel MPI support:

.. code-block:: bash

  pip install bsb[parallel]

Simulator backends
==================

If you would like to install the scaffold builder for point neuron simulations with
NEST or multi-compartmental neuron simulations with NEURON or Arbor use:

.. code-block:: bash

  pip install bsb[nest]
  # or
  pip install bsb[arbor]
  # or
  pip install bsb[neuron]
  # or any combination
  pip install bsb[arbor,nest,neuron]

.. warning::

  The NEST simulator is not installed with the `bsb-nest` package and should be set up separately.
  It installs the Python tools that the BSB needs to interface NEST. Install NEST following to their
  `installation instructions <https://nest-simulator.readthedocs.io/en/stable/installation/index.html>`_ .

Developer installation
======================

.. include:: ../dev/installation.rst
    :start-after: start-dev-install

If you want to have more information about our development guidelines, please read our
:doc:`developer guides</dev/dev-toc>`

What is next
============

You can start learning about the BSB by reading the :doc:`Getting Started section <top-level-guide>`