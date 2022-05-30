###################
Configuration files
###################

A configuration file describes the components of a scaffold model. It contains the
instructions to place and connect neurons, how to represent the cells and connections as
models in simulators and what to stimulate and record in simulations.

The default configuration format is JSON and a standard configuration file is structured
like this:

.. include:: _empty_root_nodes.rst

The :guilabel:`regions`, :guilabel:`partitions`, :guilabel:`cell_types`,
:guilabel:`placement` and :guilabel:`connectivity` spaceholders hold the configuration for
:class:`Regions <.topology.region.Region>`, :class:`Partitions
<.topology.partition.Partition>`, :class:`CellTypes <.cell_types.CellType>`,
:class:`PlacementStrategies <.placement.strategy.PlacementStrategy>` and
:class:`ConnectionStrategies <.connectivity.strategy.ConnectionStrategy>` respectively.

When you're configuring a model you'll mostly be using configuration :ref:`attributes
<config_attrs>`, :ref:`nodes <config_nodes>`, :ref:`dictionaries <config_dict>`, and
:ref:`lists <config_list>`. These configuration units can be declared through the config
file, or programatically added.

The configuration file passed the data to Python classes that use it to execute tasks such
as placing cells, connecting cells or simulating cells. In order to link your Python
classes to the configuration file they should be **importable**. The Python `documentation
<https://docs.python.org/3/tutorial/modules.html>`_ explains what modules are.

In short,  ``my_file.py`` is importable as ``my_file`` when it is in the working directory
or on the path of Python. Any classes inside of it can be referenced in a config file as
``my_file.MyClass``. Although this basic use works fine for a single directory, we have a
:doc:`best practices guide </guides/packaging>` on how to properly make your classes
discoverable on your entire machine. In the same way, you can distribute them as a package
to other people.

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

The framework will try to pass the additional keys ``value1`` and ``thingy2`` to the
class. The class should be decorated as a configuration node for it to correctly receive
and handle the values:

.. code-block:: python

  from bsb import config
  from bsb.connectivity import ConnectionStrategy

  @config.node
  class MySpecialConnection(ConnectionStrategy):
    value1 = config.attr(type=int)
    thingy2 = config.list(type=int, size=2, required=True)

For more information on creating your own configuration nodes see :doc:`nodes`.

JSON
####

The BSB uses a JSON parser with some extras. The parser has 2 special mechanisms, JSON
references and JSON imports. This allows parts of the configuration file to be reusable
across documents and to compose the document from prefab blocks. For example, an entire
simulation protocol could be imported and the start and stop time of a stimulus adjusted:

.. code-block:: json

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

This would import ``/simulations/twin_pulse`` from the ``premade_simulations.json`` JSON
document and overwrite the ``start`` and ``stop`` time of the ``pulse1`` device.

See :doc:`/config/parsers/json` to read more on the JSON parser.

.. toctree::
  :hidden:

  parsers/json

.. _default-config:

Default configuration
#####################

You can create a default configuration by calling :meth:`Configuration.default
<.config.Configuration.default>`. It corresponds to the following JSON:

.. code-block:: json

  {
    "storage": {
      "engine": "hdf5"
    },
    "network": {
      "x": 200, "y": 200, "z": 200
    },
    "partitions": {

    },
    "cell_types": {

    },
    "placement": {

    },
    "connectivity": {

    }
  }
