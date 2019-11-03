##########
Cell types
##########

Cell types are the main component of the scaffold. They will be placed into the
simulation volume and connected to eachother.

*************
Configuration
*************

In the root node of the configuration file the ``cell_types`` dictionary configures
all the cell types. The key in the dictionary will become the cell type
name. Each entry should contain a correct configuration for a PlacementStrategy
and Morphology under the ``placement`` and ``morphology`` attributes respectively.

Optionally a ``plotting`` dictionary can be provided when the scaffold's plotting
functions are used.

Basic usage
===========

1. Configure the following attributes in ``placement``:

  * ``class``: the importable name of the placement strategy class. 2 built-in
    implementation of the placement strategy are available:
    :class:`.placement.LayeredRandomWalk` and :class:`.placement.ParallelArrayPlacement`
  * ``layer``: The topological layer in which this cell type appears.
  * ``soma_radius``: Radius of the cell soma in µm.
  * ``density``: Cell density, see :ref:`specifying_cell_count` for more possibilities.

2. Select one of the morphologies that suits your cell type and configure its required
attributes.

3. The cell type will now be placed, but you'll need to configure connection types
to connect it to other cells.

Example
-------

.. code-block:: json

  {
    "cell_types": {
      "granule_cell": {
        "placement": {
          "class": "scaffold.placement.LayeredRandomWalk",
          "layer": "granular_layer",
          "soma_radius": 2.5,
          "density": 3.9e-3,
          "distance_multiplier_min": 0.5,
          "distance_multiplier_max": 0.5
        },
        "morphology": {
          "class": "scaffold.morphologies.GranuleCellGeometry",
          "pf_height": 180,
          "pf_height_sd": 20,
          "pf_length": 3000,
          "pf_radius": 0.5,
          "dendrite_length": 40
        },
        "plotting": {
          "display_name": "granule cell",
          "color": "#E62214"
        }
      }
    }
  }
