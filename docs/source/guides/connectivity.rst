============
Connectivity
============

Connection strategies connect cell types together after they've been placed into the
simulation volume. They are defined in the configuration under ``connectivity``:

.. code-block:: json

  {
    "connectivity": {
      "cell_A_to_cell_B": {
        "cls": "bsb.connectivity.VoxelIntersection",
        "pre": {
          "cell_types": ["cell_A"]
        },
        "post": {
            "cell_types": ["cell_B"]
        }
      }
    }
  }

The :guilabel:`cls` specifies which :class:`~.connectivity.strategy.ConnectionStrategy` to
load. The :guilabel:`pre` and :guilabel:`post` specify the two :class:`hemitypes
<.connectivity.strategy.HemiTypeNode>`.

Creating your own
=================

You can create custom connectivity patterns by creating an importable module (refer to the
`Python documentation <https://docs.python.org/3/tutorial/modules.html>`_) with inside a
class inheriting from :class:`~.connectivity.strategy.ConnectionStrategy`.


What follows is an example implementation, that we'll deconstruct, step by step. The
example connects cells that are near each other between a ``min`` and ``max`` distance:

.. code-block:: python

  from bsb.connectivity import ConnectionStrategy
  from bsb.exceptions import ConfigurationError
  from bsb import config
  import numpy as np
  import scipy.spatial.distance as dist

  @config.node
  class ConnectBetween(ConnectionStrategy):
    # Define the class' configuration attributes
    min = config.attr(type=float, default=0)
    max = config.attr(type=float, required=True)

    def __init__(self, **kwargs):
      # Here you can check if the object was properly configured
      if self.max < self.min:
        raise ConfigurationError("Max distance should be larger than min distance.")

    def connect(self, pre, post):
      # The `connect` function is responsible for deciding which cells get connected.
      # Use the `.placement` to get a dictionary of `PlacementSet`s to connect.
      for from_type, from_set in pre.placement.items():
        from_pos = from_set.load_positions()
        for to_type, to_set in post.placement.items():
          to_pos = to_set.load_positions()
          pairw_dist = dist.cdist(from_pos, to_pos)
          matches = (pairw_dist <= max) & (pairw_dist >= min)
          # Some more numpy code to convert the distance matches to 2 location matrices
          # ...
          pre_locs = ...
          post_locs = ...
          self.connect_cells(from_type, to_type, pre_locs, post_locs)

An example using this strategy, assuming it is importable from the ``my_module`` module:

.. code-block:: json

  {
    "connectivity": {
      "cell_A_to_cell_B": {
        "class": "my_module.ConnectBetween",
        "min": 10,
        "max": 15.5,
        "pre": {
          "cell_types": ["cell_A"]
        },
        "post": {
          "cell_types": ["cell_B"]
        }
      }
    }
  }

Then, when it is time, the framework will call the strategy's
:meth:`~.connectivity.strategy.ConnectionStrategy.connect` method.

.. rubric:: Accessing configuration values

In short, the objects that are decorated with ``@config.node`` will already be fully
configured before ``__init__`` is called and all attributes available under ``self`` (e.g.
``self.min`` and ``self.max``). For more explanation on the configuration system, see
:doc:`/config/intro`. For specifics on configuration nodes, see
:doc:`/config/module/nodes`.

.. rubric:: Accessing placement data

The ``connect`` function is handed the placement information as the ``pre`` and ``post``
parameters. The ``.placement`` attribute contains the placement data under consideration
as :class:`PlacementSets <.storage.interfaces.PlacementSet>`.

.. note::
  The ``connect`` function is called multiple times, usually once per postsynaptic "chunk"
  populated by the postsynaptic cell types. For each chunk, a region of interest is
  determined of chunks that could contain cells to be connected. This is transparent to
  you, as long as you use the ``pre.placement`` and ``post.placement`` given to you; they
  show you an encapsulated view of the placement data matching the current task. Note
  carefully that if you use the regular ``get_placement_set`` functions that they will not
  be encapsulated, and duplicate data processing might occur.

.. rubric:: Creating connections

Finally you should call ``self.scaffold.connect_cells(tag, matrix)`` to connect the cells.
The tag is free to choose, the matrix should be rows of pre to post cell ID pairs.

Connection types and labels
===========================

.. warning::
  The following documentation has not been updated to v4 yet, please bother a dev to do so
  |:stuck_out_tongue_winking_eye:|.

When defining a connection type under ``connectivity`` in the configuration file, it is
possible to select specific subpopulations inside the attributes ``from_cell_types``
and/or ``to_cell_types``. By including the attribute ``with_label`` in the
``connectivity`` configuration, you can define the subpopulation label:

.. code-block:: json

  {
    "connectivity": {
      "cell_A_to_cell_B": {
        "class": "my_module.ConnectBetween",
        "from_cell_types": [
          {
            "type": "cell_A",
            "with_label": "cell_A_type_1"
          }
        ],
        "to_cell_types": [
          {
            "type": "cell_B",
            "with_label": "cell_B_type_3"
          }
        ]
      }
    }
  }

.. note::
  The labels used in the configuration file must correspond to the labels assigned
  during cell placement.

Using more than one label
-------------------------

If under ``connectivity`` more than one label has been specified, it is possible to choose
whether the labels must be used serially or in a mixed way, by including a new attribute
``mix_labels``. For instance:

.. code-block:: json

  {
    "connectivity": {
      "cell_A_to_cell_B": {
        "class": "my_module.ConnectBetween",
        "from_cell_types": [
          {
            "type": "cell_A","with_label": ["cell_A_type_2","cell_A_type_1"]
          }
        ],
        "to_cell_types": [
          {
            "type": "cell_B","with_label": ["cell_B_type_3","cell_B_type_2"]
          }
        ]
      }
    }
  }

Using the above configuration file, the established connections are:

* From ``cell_A_type_2`` to ``cell_B_type_3``
* From ``cell_A_type_1`` to ``cell_B_type_2``

Here there is another example of configuration setting:

.. code-block:: json

  {
    "connectivity": {
      "cell_A_to_cell_B": {
        "class": "my_module.ConnectBetween",
        "from_cell_types": [
          {
            "type": "cell_A","with_label": ["cell_A_type_2","cell_A_type_1"]
          }
        ],
        "to_cell_types": [
          {
            "type": "cell_B","with_label": ["cell_B_type_3","cell_B_type_2"]
          }
        ],
        "mix_labels": true,
      }
    }
  }

In this case, thanks to the ``mix_labels`` attribute,the established connections are:

* From ``cell_A_type_2`` to ``cell_B_type_3``
* From ``cell_A_type_2`` to ``cell_B_type_2``
* From ``cell_A_type_1`` to ``cell_B_type_3``
* From ``cell_A_type_1`` to ``cell_B_type_2``
