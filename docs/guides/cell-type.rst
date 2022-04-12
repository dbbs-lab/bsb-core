##########
Cell types
##########

Cell types are the main component of the scaffold. They will be placed into the
simulation volume and connected to each other.

*************
Configuration
*************

In the root node of the configuration file the ``cell_types`` dictionary configures all
the cell types. The key in the dictionary will become the cell type name. Each entry
should contain a correct configuration for a
:class:`~.placement.strategy.PlacementStrategy` and :class:`.morphologies.Morphology`
under the ``placement`` and ``morphology`` attributes respectively.

Optionally a ``plotting`` dictionary can be provided when the scaffold's plotting
functions are used.

Basic usage
===========

1. Configure the following attributes in ``placement``:

* ``class``: the importable name of the placement strategy class. 3 built-in
  implementations of the placement strategy are available:
  :class:`~.placement.particle.ParticlePlacement`,
  :class:`~.placement.arrays.ParallelArrayPlacement` and
  :class:`~.placement.satellite.Satellite`
* ``layer``: The topological layer in which this cell type appears.
* ``soma_radius``: Radius of the cell soma in µm.
* ``density``: Cell density.

2. Select one of the morphologies that suits your cell type and configure its required
attributes. Inside of the morphology attribute, a ``detailed_morphologies`` attribute
can be specified to select detailed morphologies from the morphology repository.

3. The cell type will now be placed whenever the scaffold is compiled, but you'll need to
configure connection types to connect it to other cells.

Example
-------

.. code-block:: json

  {
    "name": "My Test configuration",
    "output": {
      "format": "bsb.output.HDF5Formatter"
    },
    "network_architecture": {
      "simulation_volume_x": 400.0,
      "simulation_volume_z": 400.0
    },
    "partitions": {
      "granular_layer": {
        "origin": [0.0, 0.0, 0.0],
        "thickness": 150
      }
    },
    "cell_types": {
      "granule_cell": {
        "placement": {
          "class": "bsb.placement.ParticlePlacement",
          "layer": "granular_layer",
          "soma_radius": 2.5,
          "density": 3.9e-3
        },
        "morphology": {
          "class": "bsb.morphologies.GranuleCellGeometry",
          "pf_height": 180,
          "pf_height_sd": 20,
          "pf_length": 3000,
          "pf_radius": 0.5,
          "dendrite_length": 40,
          "detailed_morphologies": ["GranuleCell"]
        },
        "plotting": {
          "display_name": "granule cell",
          "color": "#E62214"
        }
      }
    },
    "connectivity": {},
    "simulations": {}
  }

Use ``bsb -c=my-config.json compile`` to test your configuration file.
