############
Introduction
############

A configuration file describes a scaffold model. It contains the instructions to place and
connect neurons, how to represent the cells and connections as models in simulators and
what to stimulate and record in simulations.

The default configuration format is JSON and a standard configuration file might look like
this:

.. include:: _empty_root_nodes.rst

The :guilabel:`regions`, :guilabel:`partitions`, :guilabel:`cell_types`,
:guilabel:`placement` and :guilabel:`connection_types` spaceholders hold the configuration
for :class:`Regions <.topology.Region>`, :class:`Partitions <.topology.Partition>`,
:class:`CellTypes <.objects.CellType>`, :class:`PlacementStrategies
<.placement.PlacementStrategy>` and :class:`ConnectionStrategies
<.connectivity.ConnectionStrategy>` respectively.

When you're configuring a model you'll mostly be using **configuration attributes**,
**configuration nodes/dictionaries** and **configuration lists**. These basic concepts and
their JSON expressions are explained in :ref:`configuration-units`.

The main goal of the configuration file is to provide data to Python classes that execute
certain tasks such as placing cells, connecting them or simulating them. In order to link
your Python classes to the configuration file they should be **importable**. The Python
`documentation <https://docs.python.org/3/tutorial/modules.html>`_ explains what modules
are and are a great starting point.

In short,  ``my_file.py`` is importable as ``my_file`` when it is in the working
directory. Any classes inside of it can be referenced in a config file as
``my_file.MyClass``. Although this basic use works fine in 1 directory we have a
:doc:`best practices guide </packaging>` on how to properly make your classes discoverable
on your entire machine. You can even distribute them as a package to other people.

Here's an example of how you could use the ``MySpecialConnection`` class in your Python
file ``connectome.py`` as a class in the configuration:

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

Any extra configuration data (such as ``value1`` and ``thingy2``) is automatically passed
to it!

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
