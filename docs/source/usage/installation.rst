==================
Installation Guide
==================

Preamble
========



.. warning::

	Your mileage with the framework will vary based on your adherence to Python best
	practices.

Which Python to use?
--------------------

Linux distributions come bundled with Python installations and many parts of the
distro depend on these installations, making them hard to update and installing
packages into the system-wide environment can have surprising side effects.

Instead to stay up to date with the newest Python releases use a tool like
`pyenv <https://github.com/pyenv/pyenv#simple-python-version-management-pyenv>`_
to manage different Python versions at the same time. Windows users can install
a newer binary from the Python website. You're also most likely to make a big
bloated mess out of these environments and will run into myriads of strange
environment errors.

Why is everyone telling me to use a virtual env?
------------------------------------------------

Python's package system is flawed, it can only install packages in a "global"
fashion. You can't install multiple versions of the same package for different
projects so eventually packages will start clashing with each other. On top of
that scanning the installed packages for metadata, like plugins, becomes slower
the more packages you have installed.

To fix these problems Python relies on "virtual environments". Use either
``pyenv`` (mentioned above), ``venv`` (part of Python's stdlib) or if you must
``virtualenv`` (package). Packages inside a virtual environment do not clash
with packages from another environment and let you install your dependencies on
a per project basis.

Instructions
============

The scaffold framework can be installed using ``pip``:

  .. code-block:: bash

    pip install bsb

You can verify that the installation works with

.. code-block:: bash

    bsb -v=3 compile -x=100 -z=100 -p

This should generate a template config and an HDF5 file in your current directory and open
a plot of the generated network, it should contain a column of ``base_type`` cells. If no
errors occur you are ready to :doc:`get started <getting-started>`.

Another verification method is to import the package in a Python script:

.. code-block:: python

  from bsb.core import Scaffold

  # Create an empty scaffold network with the default configuration.
  scaffold = Scaffold()

Simulator backends
==================

If you'd like to install the scaffold builder for point neuron simulations with
NEST or multicompartmental neuron simulations with NEURON use:

.. code-block:: bash

  pip install bsb[nest]
  # or
  pip install bsb[neuron]
  # or both
  pip install bsb[nest,neuron]

.. note::

	This does not install the simulators, just the Python requirements for the
	framework to handle simulations using these backends.


Installing for NEURON
=====================

The BSB's installation will install NEURON from PyPI if no ``NEURON`` installation is
detected by ``pip``. This means that any custom installations that rely on ``PYTHONPATH``
to be detected at runtime but aren't registered as an installed package to pip will be
overwritten. Because it is quite common for NEURON to be incorrectly installed from pip's
point of view, you have to explicitly ask the BSB installation to install it:

.. code-block:: bash

    pip install bsb[neuron]

After installation of the dependencies you will have to describe your cell models using
`Arborize's <https://arborize.readthedocs.io>`_ ``NeuronModel`` template and import your
Arborize cell models module into a ``MorphologyRepository``:

.. code-block:: bash

    $ bsb
    > open mr morphologies.hdf5 --create
    <repo 'morphologies.hdf5'> arborize my_models
    numprocs=1
    Importing MyCell1
    Importing MyCell2
    ...
    <repo 'morphologies.hdf5'> exit

This should allow you to use ``morphologies.hdf5`` and the morphologies contained within
as the :guilabel:`morphology_repository` of the :guilabel:`storage` node in your config:

.. code-block:: json

    {
        "name": "Example config",
        "storage": {
            "engine": "hdf5",
            "root": "my_network.hdf5",
            "morphology_repository": "morphologies.hdf5"
        }
    }



Installing NEST
===============

The BSB currently runs a fork of NEST 2.18, to install it, follow the instructions below.
The instructions assume you are using `pyenv`_
for virtual environments.

.. code-block:: bash

  sudo apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
  git clone git@github.com:dbbs-lab/nest-simulator
  cd nest-simulator
  mkdir build && cd build
  export PYTHON_CONFIGURE_OPTS="--enable-shared"
  # Any Python 3.8+ version built with `--enable-shared` will do
  PYVER_M=3.9
  PYVER=$PYVER_M.0
  VENV=nest-218
  pyenv install $PYVER
  pyenv virtualenv $PYVER $VENV
  pyenv local nest-218
  cmake .. \
    -DCMAKE_INSTALL_PREFIX=$(pyenv root)/versions/$VENV \
    -Dwith-mpi=ON \
    -Dwith-python=3 \
    -DPYTHON_LIBRARY=$(pyenv root)/versions/$PYVER/lib/libpython$PYVER_M.so \
    -DPYTHON_INCLUDE_DIR=$(pyenv root)/versions/$PYVER/include/python$PYVER_M
  make install -j8

Confirm your installation with:

.. code-block:: bash

  python -c "import nest; nest.test()"

.. note::

	There might be a few failed tests related to ``NEST_DATA_PATH`` but this is OK.
