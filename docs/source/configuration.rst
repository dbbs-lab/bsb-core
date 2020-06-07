#######################
Configuration reference
#######################

.. note::
  The key of a configuration object in its parent will be stored as its :guilabel:`name`
  property and is used throughout the package. Some of these values are
  hardcoded into the package and the names of the standard configuration objects
  should not be changed.

====================
Role in the scaffold
====================

Configuration plays a key role in the scaffold builder. It is the main mechanism to
describe a model. A scaffold model can be initialized from a Configuration object, either
from a standalone file or provided by the StorageEngine. In both cased the raw
configuration string is parsed into a Python tree of dictionaries and lists. This
configuration tree is then passed to the Configuration class for casting. How a tree is to
be cast into a Configuration object can be described using configuration unit syntax.

===================
Configuration units
===================

When the configuration tree is being cast into a Configuration object there are 4 key units:

- A **configuration attribute** represented by a JSON key-value pair.
- A **configuration node** represented by a JSON dictionary where each key-value pair
  represents an attribute of the node.
- A **configuration dictionary** represented by a JSON dictionary where each key-value pair
  represents another configuration unit.
- A **configuration list** represented by a JSON list where each key-value pair represents
  another configuration unit.

.. note::

  If a JSON list or dictionary contains regular values instead of configuration units, the
  :class:`types.list <.config.types.list>` and :class:`types.dict <.config.types.dict>`
  are used instead of the :class:`conf.list <.config.list>` and
  :class:`conf.dict <.config.dict>`.

Configuration nodes
===================

A node in the configuration can be described by creating a class and applying the ``@config.node`` decorator to it.
This decorator will look for ``config.attr`` and other configuration unit constructors on the class
to create the configuration information on the class. This node class can then be used in
the type argument of another configuration attribute, dictionary, or list:

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyNode:
    name = config.attr(type=str, required=True)
    sweetness = config.attr(type=float, default=3.0)

This candy node class now represents the following JSON dictionary:

.. code-block:: json

  {
    "name": "Lollypop",
    "sweetness": 12.0
  }

You will mainly design configuration nodes and other configuration logic when designing
custom strategies.

Configuration attributes
========================

An attribute can refer to a singular value of a certain type, or to another node:

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyStack:
    count = config.attr(type=int, required=True)
    candy = config.attr(type=CandyNode)

.. code-block:: json

  {
    "count": 12,
    "candy": {
      "name": "Hardcandy",
      "sweetness": 4.5
    }
  }

Configuration dictionaries
==========================

Configuration dictionaries hold configuration nodes. If you need a dictionary of values
use the :class:`types.dict <.config.types.dict>` syntax instead.

.. code-block:: python

  from scaffold import config

  @config.node
  class CandyNode:
    name = config.attr(key=True)
    sweetness = config.attr(type=float, default=3.0)

  @config.node
  class Inventory:
    candies = config.dict(type=CandyStack)

.. code-block:: json

  {
    "candies": {
      "Lollypop": {
        "sweetness": 12.0
      },
      "Hardcandy": {
        "sweetness": 4.5
      }
    }
  }

Items in configuration dictionaries can be accessed using dot notation or indexing:

.. code-block:: python

  inventory.candies.Lollypop == inventory.candies["Lollypop"]

Using the ``key`` keyword argument on a configuration attribute will pass the key in the
dictionary to the attribute so that ``inventory.candies.Lollypop.name == "Lollypop"``.

Configuration lists
===================

Configuration dictionaries hold unnamed collections of configuration nodes. If you need a
list of values use the :class:`types.list <.config.types.list>` syntax instead.

.. code-block:: python

  from scaffold import config

  @config.node
  class Inventory:
    candies = config.list(type=CandyStack)

.. code-block:: json

  {
    "candies": [
      {
        "count": 100,
        "candy": {
          "name": "Lollypop",
          "sweetness": 12.0
        }
      },
      {
        "count": 1200,
        "candy": {
          "name": "Hardcandy",
          "sweetness": 4.5
        }
      }
    ]
  }

===============
Root attributes
===============

The root node accepts the following attributes:

* :guilabel:`name`: *Unused*, a name for the configuration file. Is stored in the output
  files so it can be used for reference.
* :guilabel:`storage`: Configuration node for the storage engine.
* :guilabel:`network`: Configuration node for general simulation properties.
* :guilabel:`layers`: Configuration dictionary for :class:`Layers <.objects.Layer>`.
* :guilabel:`cell_types`: A dictionary containing the :class:`.models.CellType` configurations.
* :guilabel:`connection_types`: A dictionary containing the :class:`.connectivity.ConnectionStrategy` configurations.
* :guilabel:`simulations`: A dictionary containing the :class:`.simulation.SimulationAdapter` configurations.

.. code-block:: json

  {
    "name": "...",
    "output": {

    },
    "network_architecture": {

    },
    "layers": {
      "some_layer": {

      },
      "another_layer": {

      }
    },
    "cell_types": {

    },
    "connection_types": {

    },
    "simulations": {

    }
  }

=================
Output attributes
=================

Format
======

This attribute is a string that refers to the implementation of the Storage
that should be used::

  {
    "output": {
      "format": "scaffold.output.HDF5Formatter"
    }
  }

If you write your own implementation the string should be discoverable by Python.
Here is an example for ``MyStorage`` in a package called ``my_package``::

  {
    "output": {
      "format": "my_package.MyStorage"
    }
  }

Your own implementations must inherit from :class:`.output.Storage`.

File
====

Determines the path and filename of the output file produced by the output
formatter. This path is relative to Python's current working directory.

::

  {
    "output": {
      "file": "my_file.hdf5"
    }
  }

===============================
Network architecture attributes
===============================

simulation_volume_x
===================

The size of the X dimension of the simulation volume.

simulation_volume_z
===================

The size of the Z dimension of the simulation volume.

::

  {
    "network_architecture": {
      "simulation_volume_x": 150.0,
      "simulation_volume_z": 150.0
    }
  }

.. note::
  The Y can not be set directly as it is a result of stacking/placing the layers.
  It's possible to place cells outside of the simulation volume, and even to place
  layers outside of the volume, but it is not recommended behavior. The X and Z
  size are merely the base/anchor and a good indicator for the scale of the
  simulation, but they aren't absolute restrictions.

.. warning::
  Do not modify these values directly on the configuration object: It will not
  rescale your layers. Use :func:`resize <.configuration.ScaffoldConfig.resize>` instead.

================
Layer attributes
================

position
========

*(Optional)* The XYZ coordinates of the bottom-left corner of the layer. Is overwritten if
this layer is part of a `stack`_.

::

  "some_layer": {
    position: [100.0, 0.0, 100.0]
  }

thickness
=========

A fixed value of Y units.

Required unless the layer is scaled to other layers.

::

  "some_layer": {
    "thickness": 600.0
  }

.. _cref_xz_scale:

xz_scale
========

*(Optional)* The scaling of this layer compared to the simulation volume. By
default a layer's X and Z scaling are ``[1.0, 1.0]`` and so are equal to the
simulation volume.

::

  "some_layer": {
    "xz_scale": [0.5, 2.0]
  }

xz_center
=========

*(Optional)* Should this layer be aligned to the corner or the center of the
simulation volume? Defaults to ``False``.

.. _cref_stack:

stack
=====

*(Optional)* Layers can be stacked on top of eachother if you define this attribute and
give their stack configurations the same :guilabel:`stack_id`. The
:guilabel:`position_in_stack` will determine in which order they are stacked, with the
lower values placed on the bottom, receiving the lower Y coordinates. Exactly one layer
per stack should define a :guilabel:`position` attribute in their stack configuration to
pinpoint the bottom-left corner of the start of the stack.

.. _cref_stack_id:

stack_id
--------

Unique identifier of the stack. All layers with the same stack id are grouped together.

.. _cref_position_in_stack:

position_in_stack
-----------------

Unique identifier for the layer in the stack. Layers with larger positions will be placed
on top of layers with lower ids.

.. _cref_stack_position:

position
--------

This attribute needs to be specified in exactly one layer's :guilabel:`stack` dictionary
and determines the starting (bottom-corner) position of the stack.

Example
-------

This example defines 2 layers in the same stack::

  {
    "layers": {
      "top_layer": {
      "thickness": 300,
        "stack": {
          "stack_id": 0,
          "position_in_stack": 1,
          "position": [0., 0., 0.]
        }
      },
      "bottom_layer": {
        "thickness": 200,
        "stack": {
          "stack_id": 0,
          "position_in_stack": 0
        }
      }
    }
  }

.. _cref_volume_scale:

volume_scale
============

*(Optional)* The scaling factor used to scale this layer with respect to other layers. If
this attribute is set, the :guilabel:`scale_from_layers` attribute is also required.

::

  "some_layer": {
    "volume_scale": 10.0,
    "scale_from_layers": ["other_layer"]
  }

.. _cref_scale_from_layers:

scale_from_layers
=================

*(Optional)* A list of layer names whose volume needs to be added up, and this layer's
volume needs to be scaled to.

Example
-------

Layer A has a volume of ``2000.0``, Layer B has a volume of ``3000.0``.
Layer C specifies a :guilabel:`volume_scale` of ``10.0`` and :guilabel:`scale_from_layers` = ``["layer_a",
"layer_b"]``; this will cause it to become a cube (unless `volume_dimension_ratio`_ is
specified) with a volume of ``(2000.0 + 3000.0) * 10.0 = 50000.0``

.. _cref_volume_dimension_ratio:

volume_dimension_ratio
======================

*(Optional)* Ratio of the rescaled dimensions. All given numbers are normalized to the Y
dimension::

  "some_layer": {
    "volume_scale": 10.0,
    "scale_from_layers": ["other_layer"],
    # Cube (default):
    "volume_dimension_ratio": [1., 1., 1.],
    # High pole:
    "volume_dimension_ratio": [1., 20., 1.], # Becomes [0.05, 1., 0.05]
    # Flat bed:
    "volume_dimension_ratio": [20., 1., 20.]
  }

====================
Cell Type Attributes
====================

entity
======

If a cell type is marked as an entity with ``"entity": true``, it will not receive a
position in the simulation volume, but it will still be assigned an ID during placement
that can be used for the  connectivity step. This is for example useful for afferent
fibers.

If :guilabel:`entity` is ``true`` no :guilabel:`morphology` or :guilabel:`plotting` needs
to be specified.

relay
=====

If a cell type is a :guilabel:`relay` it immediately relays all of its inputs to its
target cells. Also known as a parrot neuron.

placement
=========

Configuration node of the placement of this cell type. See :ref:`cref_placement`.

morphology
==========

Configuration node of the morphologies of this cell type. This is still an experimental
API, expect changes. See :ref:`cref_morphology`.

plotting
========

Configuration node of the plotting attributes of this cell type. See :ref:`cref_plotting`.

Example
=======

.. code-block::



.. _cref_placement:

====================
Placement Attributes
====================

Each configuration node needs to specify a :class:`.placement.PlacementStrategy` through
:guilabel:`class`. Depending on the strategy another specific set of attributes is
required. To see how to configure each :class:`.placement.PlacementStrategy` see the
:doc:`guides/placement-strategies`.

class
=====

A string containing a PlacementStrategy class name, including its module.

.. code-block::

  "class": "scaffold.placement.ParticlePlacement"

=======================
Connectivity Attributes
=======================

The connectivity configuration node contains some basic attributes listed below and a set
of strategy specific attributes that you can find in
:doc:`guides/connection-strategies`.

class
=====

A string containing a ConnectivityStrategy class name, including its module.

.. code-block::

  "class": "scaffold.placement.VoxelIntersection"

from_types/to_types
===================

A list of pre/postsynaptic selectors. Each selector is made up of a :guilabel:`type` to
specify the cell type and a :guilabel:`compartments` list that specify the involved
compartments for morphologically detailed connection strategies.

.. deprecated:: 4.0

  Each connectivity type will only be allowed to have 1 presynaptic and postsynaptic cell
  type. :guilabel:`from/to_types` will subsequently be renamed to :guilabel:`from/to_type`

.. code-block::

  "from_types": [
    {
      "type": "example_cell",
      "compartments": [
        "axon"
      ]
    }
  ]

.. _cref_morphology:

=====================
Morphology attributes
=====================

.. _cref_plotting:

===================
Plotting attributes
===================

color
=====

The color representation for this cell type in plots. Can be any valid Plotly
color string.

.. code-block::

  "color": "black"
  "color": "#000000"

label
=====

The legend label for this cell type in plots.

.. code-block::

  "label": "My Favourite Cells"
