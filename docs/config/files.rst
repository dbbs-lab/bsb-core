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
<config_attrs>`, :ref:`nodes <config_nodes>`, :ref:`dictionaries <config_dict>`,
:ref:`lists <config_list>`, and :ref:`references <config_ref>`. These configuration units
can be declared through the config file, or programatically added.

Code
####

Most of the framework components pass the data on to Python classes, that determine the
underlying code strategy of the component. In order to link your Python classes to the
configuration file they should be an `importable module
<https://docs.python.org/3/tutorial/modules.html>`_. Here's an example of how the
``MySpecialConnection`` class in the local Python file ``connectome.py`` would be
available to the configuration:

.. code-block:: json

  {
    "connectivity": {
      "A_to_B": {
        "strategy": "connectome.MySpecialConnection",
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

The BSB uses a JSON parser with some extras. The parser has 2 special mechanisms,
:ref:`JSON references <json_ref>` and :ref:`JSON imports <json_import>`. This allows parts
of the configuration file to be reusable across documents and to compose the document from
prefab blocks.

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
