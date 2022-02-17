
!!!!!!!!!  CURRENTLY IN EXCLUDEPATTERNS  !!!!!!!!!
!!   This file is ignored while we restructure  !!
!!             the documentation                !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


######
Layers
######

Layers are partitions of the simulation volume that most placement strategies use as a
reference to place cells in.

*************
Configuration
*************

In the root node of the configuration file the ``layers`` dictionary configures all the
layers. The key in the dictionary will become the layer name. A layer configuration
requires only to describe its origin and dimensions. In its simplest form this can be
achieved by providing a ``position`` and ``thickness``. In that case the layer will scale
along with the simulation volume ``X`` and ``Z``.

Basic usage
===========

Configure the following attributes:

* ``position``: XYZ coordinates of the bottom-left corner, unless ``xz_center`` is set.
* ``thickness``: Height of the layer

Example
-------

.. code-block:: json

  {
    "layer": {
      "granular_layer": {
        "position": [0.0, 600.0, 0.0],
        "thickness": 150.0
      }
    }
  }

Stacking layers
===============

Placing layers manually can be sufficient, but when you have layers with dynamic sizes it
can be usefull to automatically rearrange other layers. To do so you can group layers
together in a vertical stack. To stack layers together you need to configure
:ref:`cref_stack` dictionaries in both with the same :ref:`cref_stack_id` and different
:ref:`cref_position_in_stack`. Each stack requires exactly one definition of its
:ref:`cref_stack_position`, which can be supplied in any of the layers it consists of::

  "partitions": {
    "layer_a": {
      "thickness": 150.0,
      "stack": {
        "stack_id": 0,
        "position_in_stack": 0,
        "position": [10, 0, 100]
      }
    },
    "layer_b": {
      "thickness": 150.0,
      "stack": {
        "stack_id": 0,
        "position_in_stack": 1
      }
    }
  }

This will result in a stack of Layer A and B with Layer B on top. Both layers will
have an X and Z origin of ``10`` and ``100``, but the Y of Layer B will be raised from
``0`` with the thickness of Layer A, to ``150``, ending up on top of it. Both Layer A and
B will have X and Z dimensions equal to the simulation volume X and Z. This can be altered
by specifying :ref:`cref_xz_scale`.

Scaling layers
==============

Layers by default scale with the simulation volume X and Z. You can change the default
one-to-one ratio by specifying :ref:`cref_xz_scale`::

  "layer_a": {
    "xz_scale": 0.5
  }

When the XZ size is ``[100, 100]`` layer A will be ``[50, 50]`` instead. You can also use
a list to scale different on the X than on the Z axis::

  "layer_a": {
    "xz_scale": [0.5, 2.0]
  }

Volumetric scaling
------------------

Layers can also scale relative to the volume of other layers. To do so set a
:ref:`cref_volume_scale` ratio which will determine how many times larger the volume of
this layer will be than its reference layers. The reference layers can be specified with
:ref:`cref_scale_from_layers`. The shape of the layer will be cubic, unless the
:ref:`cref_volume_dimension_ratio` is specified::

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

.. note::

  The ``volume_dimension_ratio`` is normalized to the Y value.

*********
Scripting
*********

The value of layers in scripting is usually limited because they only contain spatial
information.

Retrieving layers
=================

Layers can be retrieved from a :class:`ScaffoldConfig <.config.ScaffoldConfig>`:

.. code-block:: python

  from bsb.config import JSONConfig

  config = JSONConfig("mouse_cerebellum")
  layer = config.get_layer(name="granular_layer")

A :class:`Scaffold <.core.Scaffold>` also stores its configuration:

.. code-block:: python

  layer = scaffold.configuration.get_layer(name="granular_layer")

All :class:`Layered <.placement.Layered>` placement strategies store a reference to their layer
instance:

.. code-block:: python

  placement = scaffold.get_cell_type("granule_cell").placement
  layer_name = placement.layer
  layer = placement.layer_instance

.. note::

  The instance of a placement strategy's layer is added only after initialisation of the
  placement strategy, which occurs only after the scaffold is bootstrapped (so after
  ``scaffold = Scaffold(config)``)
