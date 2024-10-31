#######
Regions
#######

===========
RegionGroup
===========
:class:`RegionGroup <.topology.region.RegionGroup>` is the simplest implementation
of the :class:`Region class <.topology.region.Region>`. It serves as a container
interface for its children. Any transformation operation applied to it will be directly
applied to all its children independently.

Parameters
----------
* ``name``: Name of the `RegionGroup` region.
* ``children``: Reference to Regions or Partitions belonging to this region.


=====
Stack
=====
:class:`Stack <.topology.region.Stack>` groups its children on top of each other along a
defined axis (`main axis`) and adjust its length accordingly.

This class first place the first of its children at its defined coordinates. Then, each
following child of the `Stack` will see their `main axis` coordinates set according to
its predecessor resolved position and thickness along the `main axis`. The user can also
define a reference child as the origin of the stack (the coordinates will be relative to
this child). The order of the ``children`` field in the configuration corresponds to the
final order of the `Stack`.

Parameters
----------

* ``name``: Name of the `Stack` region.
* ``axis``: `Main axis` along which the `Stack`'s children will be stacked. Should be one
  of ["x", "y", "z"].
* ``anchor``: Reference to one child of the stack, which origin will become the origin of
  the stack.
* ``children``: Reference to Regions or Partitions belonging to this region.

.. note::

    `Stack` is mainly meant to contain `Layer` partitions. Indeed, if a `Stack` and its
    `Layers` share the same ``axis``, then each `Layer` will occupy the whole space of the
    Region, except on the ``axis`` where it will be defined according to their ``thickness``.

Example
-------
Here we create a `Stack` of `Layers` along the `y` axis with the ``top_layer`` as anchor:

.. tab-set-code::

    .. code-block:: json

        {
          "regions": {
            "brain_region": {
              "type": "stack",
              "children": ["bottom_layer", "top_layer"],
              "anchor": "top_layer",
              "axis": "y",
            }
          },
          "partitions": {
            "bottom_layer": {
              "type": "layer",
              "thickness": 50,
            },
            "top_layer": {
              "type": "layer",
              "thickness": 100,
            }
          },
        }

    .. code-block:: python

        from bsb import Stack, Layer

        top_layer = Layer(name="top_layer", thickness=100, axis="y")
        bottom_layer = Layer(name="bottom_layer", thickness=50, axis="y")
        stack = Stack(name="stack", children=["bottom_layer","top_layer"], axis="y")

This means that the ``top_layer`` and ``bottom_layer`` will occupy all space available to
``brain_region`` along the `x` and `z` axis.

Because ``top_layer``  is the anchor of ``brain_region`` (despite being the second defined
in the ``children`` field of ``brain_region``), its origin will be (0, 0, 0) will spread
between [0; 100] along the `y` axis.
The ``bottom_layer`` arrives before ``top_layer`` in the ``children`` order so its origin
will be (0, -50, 0) will spread between [-50; 0] along the `y` axis.
