############################
List of placement strategies
############################

*****************
LayeredRandomWalk
*****************

*Class*: :class:`.placement.LayeredRandomWalk`

The LayeredRandomWalk class will sudivide the layer volume into sublayers and divide
cells among them. In each sublayer a self-avoiding random walk will be performed
to distribute the cells throughout the sublayer. A random yitter is applied in
height and the sublayers are stacked on top of eachother.

Configuration
=============

.. include:: ./standard-placement-attributes.rst

* ``y_restriction`` *(optional)*: a 2 element array that can restrict placement
  to within certain y-axis bounds inside of a layer. For example ``y_restriction =
  [0.5, 1.0]`` will restrict placement to the top half of the layer.
* ``distance_multiplier_min/max``: These are multiplied by the :ref:`cell epsilon
  <cell_epsilon>` to
  obtain minimum and maximum distances between a newly placed and the previous
  cell. Default values are 0.75 and 1.25.

**********************
ParallelArrayPlacement
**********************

*Class*: :class:`.placement.ParallelArrayPlacement`
