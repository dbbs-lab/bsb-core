#######################
Configuration reference
#######################

.. note::
  The key of a configuration object in its parent will be stored as its :guilabel:`name`
  property and is used throughout the package. Some of these values are
  hardcoded into the package and the names of the standard configuration objects
  should not be changed.

===============
Root attributes
===============

The root node accepts the following attributes:

* :guilabel:`name`: *Unused*, a name for the configuration file. Is stored in the output
  files so it can be used for reference.
* :guilabel:`output`: Configuration object for the output :class:`.output.HDF5Formatter`.
* :guilabel:`network_architecture`: Configuration object for general simulation properties.
* :guilabel:`layers`: A dictionary containing the :class:`.models.Layer` configurations.
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
