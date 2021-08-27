############
Introduction
############

A configuration file describes a scaffold model. It contains the instructions to place and
connect neurons, how to represent the cells and connections as models in simulators and
what to stimulate and record in simulations.

The default configuration format is JSON and a standard configuration file might look like
this:

.. code-block:: json

  {
    "storage": {
      "engine": "hdf5",
      "root": "my_network.hdf5"
    },
    "network": {
      "x": 200,
      "z": 200
    },
    "regions": {

    },
    "partitions": {

    },
    "cell_types": {

    },
    "connectivity": {

    }
  }

The ``regions``, ``layers``, ``cell_types`` and ``connection_types`` spaceholders would
hold configuration for :class:`Regions <.objects.Region>`, :class:`Layers
<.objects.Layer>`, :class:`CellTypes <.objects.CellType>` and :class:`ConnectionStrategies
<.connectivity.ConnectionStrategy>` respectively.

Basic use
#########

When you're configuring a model you'll mostly be using **configuration attributes**,
**configuration nodes/dictionaries** and **configuration lists**. These basic concepts and
their JSON expressions are explained in :ref:`configuration-units`.

The main goal of the configuration file is to provide data to Python classes that execute
certain tasks such as placing cells, connecting them or simulating them. In order to link
your Python classes to the configuration file they should be **importable**. The Python
`documentation <https://docs.python.org/3/tutorial/modules.html>`_ explains what modules
are and are a great starting point. In short when you're working from the same directory
any ``my_file.py`` is importable as the ``my_file`` module. Any classes inside of it can
be referenced in a config file as ``my_file.MyClass``. Although this basic use works fine
in 1 directory we have a :doc:`best practices guide </packaging>` on how to properly make
your classes folder independent, discoverable on your entire machine and even how to
distribute them as a package.

Here's an example of how you could use the ``MySpecialConnection`` class in your Python
file ``connectome.py`` as a class in the configuration. It will then be loaded and any
extra configuration data (such as ``value1`` and ``thingy2``) is passed along to it:

.. code-block:: json

  {
    "storage": {
      "engine": "hdf5",
      "root": "my_network.hdf5"
    },
    "network": {
      "x": 200,
      "z": 200
    },
    "regions": {

    },
    "partitions": {
    },
    "cell_types": {

    },
    "connectivity": {
      "A_to_B": {
        "cls": "connectome.MySpecialConnection",
        "value1": 15,
        "thingy2": [4, 13]
      }
    }
  }

For more information on creating your own configuration nodes see :doc:`module/nodes`.

JSON Parser
###########

The BSB uses a json parser with some extras. The parser has 2 special
mechanisms, JSON references and JSON imports. This allows parts of the
configuration file to be reusable across documents and to compose the document
from prefab blocks where only some key aspects are adjusted. For example, an
entire simulation protocol could be imported and the start and stop time of a
stimulus adjusted::

  {
    "simulations": {
      "premade_sim": {
        "$ref": "premade_simulations.json#/simulations/twin_pulse",
        "devices": {
          "pulse1": {
            "start": 100,
            "stop": 200
          }
        }
      }
    }
  }

This would import ``/simulations/twin_pulse`` from the
``premade_simulations.json`` JSON document and overwrite the ``start`` and
``stop`` time of the ``pulse1`` device.

See :doc:`/config/parsers/json` to read more on the JSON parser.

Network
#######

This node contains some basic properties of the network configured in this file.

The :guilabel:`x` and :guilabel:`z` attributes are loose indicators of the scale of the
simulation. You can use them to scale the volume of your layers or for other mechanisms
that determine the region in which to place your cells. They do not restrict placement,
things can still be placed outside the specified [0, x] and [0, z] region.

Basic cell types
################

For a more complete guide see :doc:`/config/cell-types`

Spatial
-------

For a more complete guide see :doc:`/config/spatial`

Placement
---------

For a more complete guide see :doc:`/config/placement-strategies`

Basic layers
############

For a more complete guide see :doc:`/config/layers`

Basic regions
#############

For a more complete guide see :doc:`/config/regions`

Basic connection strategies
###########################

For a more complete guide see :doc:`/config/connection-strategies`

Postprocessing hooks
####################

For a more complete guide see :doc:`/config/postprocessing`

After placement
---------------

After connectivity
------------------
