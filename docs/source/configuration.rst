:tocdepth: 2

#############
Configuration
#############

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  configuration/cell-type
  configuration/connection-type
  configuration/placement-strategies
  configuration/connection-strategies
  configuration/simulation

.. note::
  The key of a configuration object in its parent will be stored as its ``name``
  property and is used throughout the package. Some of these values are
  hardcoded into the package and the names of the standard configuration objects
  should not be changed.

===============
Root attributes
===============

The root node accepts the following attributes:

* ``name``: *Unused*, a name for the configuration file. Is stored in the output
  files so it can be used for reference.
* ``output``: Configuration object for the output :class:`HDF5Formatter`.
* ``network_architecture``: Configuration object for general simulation properties.
* ``layers``: A dictionary containing the :class:`Layer` configurations.
* ``cell_types``: A dictionary containing the :class:`CellType` configurations.
* ``connection_types``: A dictionary containing the :class:`ConnectionStrategy` configurations.
* ``simulations``: A dictionary containing the :class:`SimulationAdapter` configurations.

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

This attribute is a string that refers to the implementation of the OutputFormatter
that should be used::

  {
    "output": {
      "format": "scaffold.output.HDF5Formatter"
    }
  }

If you write your own implementation the string should be discoverable by Python.
Here is an example for ``MyOutputFormatter`` in a package called ``my_package``::

  {
    "output": {
      "format": "my_package.MyOutputFormatter"
    }
  }

Your own implementations must inherit from :class:`.output.OutputFormatter`.

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
  rescale your layers. Use :func:`.configuration.ScaffoldConfig.resize` instead.

================
Layer attributes
================


thickness
=========

A fixed value of Y units that this layer will be high/thick/deep.

Required unless the layer is scaled to other layers.

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

stack
=====

Layers can be stacked on top of eachother if you define this attribute and give
their stack configurations the same ``stack_id``. The ``position_in_stack`` will
determine in which order they are stacked, with the lower values placed on the
bottom, receiving the lower Y coordinates. Exactly one layer per stack should
define a ``position`` attribute in their stack configuration to pinpoint the
bottom-left corner of the start of the stack.

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
