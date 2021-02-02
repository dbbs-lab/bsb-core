=================
Istallation Guide
=================

Preamble
========

Your mileage with the framework will vary based on your adherence to Python's best
practices. Some big fat warnings:

.. error::

	DO NOT USE YOUR SYSTEM PYTHON ON LINUX.

Linux distributions come bundled with Python installations and many parts if not all parts
of Linux depend on these Python installations. Updating these installations is nigh
impossible and even installing Python packages might break your Linux box.

Instead to stay up to date with the newest Python releases use a tool like `pyenv
<https://github.com/pyenv/pyenv#simple-python-version-management-pyenv>`_ to manage
different Python versions at the same time. Windows users can simply install a newer
binary from the Python website.

.. error::

	SET UP A VIRTUAL ENVIRONMENT.

Python's package system is flawed, it can only install packages in a "global" fashion. You
can't install multiple versions of the same package for different projects so eventually
packages will start clashing with each other. On top of that scanning the installed
packages for metadata, like plugins, becomes slower the more packages you have installed.

To fix these problems Python relies on "virtual environments". Use either ``pyenv``
(mentioned above), ``venv`` (part of Python's stdlib) or if you must ``virtualenv``
(package). Packages inside a virtual environment do not clash with packages from another
environment and let you install your dependencies on a per project basis.

Instructions
============

The scaffold framework can be installed using ``pip``:

  .. code-block:: bash

    pip install bsb

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
