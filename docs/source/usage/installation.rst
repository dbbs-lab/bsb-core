=================
Istallation Guide
=================

The scaffold framework can be installed using Pip for Python 3

.. code-block:: bash

    pip install bsb

You can verify that the installation works with

.. code-block:: bash

    bsb make-config
    bsb -v=3 compile -x=50 -z=50 -p

This should generate a template config and an HDF5 file in your current directory and open
a plot of the generated network, it should contain a column of ``base_type`` cells. If no
errors occur you are ready to :doc:`get started <getting-started>`.


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
as the :guilabel:`morphology_repository` of the :guilabel:`output` node in your config:

.. code-block:: json

    {
        "name": "Example config",
        "output": {
        "format": "bsb.output.HDF5Formatter",
        "file": "my_network.hdf5",
        "morphology_repository": "morphologies.hdf5"
        }
    }
