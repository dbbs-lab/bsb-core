##################
List of strategies
##################

:class:`AllToAll <.connectivity.general.AllToAll>`
==================================================

This strategy connects each presynaptic neuron to all the postsynaptic neurons.
It therefore creates one connection for each unique pair of neuron.

:class:`FixedIndegree <.connectivity.general.FixedIndegree>`
============================================================

This strategy connects to each postsynaptic neuron, a fixed number of uniform randomly selected
presynaptic neurons.

* ``indegree``: Number of neuron to connect for each postsynaptic neuron.

.. tab-set-code::

    .. code-block:: json

           "connectivity": {
           "A_to_B": {
               "strategy": "bsb.connectivity.FixedIndegree",
               "presynaptic": {
                       "cell_types": ["A"]
               },
               "postsynaptic": {
                       "cell_types": ["B"]
               },
               "indegree": 2
           }

    .. code-block:: python

      config.connectivity.add(
        "A_to_B",
        strategy="bsb.connectivity.FixedIndegree",
        presynaptic=dict(cell_types=["A"]),
        postsynaptic=dict(cell_types=["type_B"]),
        indegree= 2
      )

.. note::
  In this example every cell of type B is connected to two cells of type A.


:class:`FixedOutdegree <.connectivity.general.FixedOutdegree>`
==============================================================

This strategy connects to each presynaptic neuron, a fixed number of uniform randomly selected
postsynaptic neurons.

* ``outdegree``: Number of neuron to connect for each presynaptic neuron.

:class:`VoxelIntersection <.connectivity.detailed.voxel_intersection.VoxelIntersection>`
========================================================================================

This strategy voxelizes morphologies into collections of cubes, thereby reducing the
spatial specificity of the provided traced morphologies by grouping multiple compartments
into larger cubic voxels. Intersections are found not between the seperate compartments
but between the voxels and random compartments of matching voxels are connected to eachother.
This means that the connections that are made are less specific to the exact morphology
and can be very useful when only 1 or a few morphologies are available to represent each
cell type.

* ``affinity``: A fraction between 1 and 0 which indicates the tendency of cells to form
  connections with other cells with whom their voxels intersect. This can be used to
  downregulate the amount of cells that any cell connects with.
* ``contacts``: A number or distribution determining the amount of synaptic contacts one
  cell will form on another after they have selected eachother as connection partners.

.. note::
  The affinity only affects the number of cells that are contacted, not the number of
  synaptic contacts formed with each cell.

.. tab-set-code::

    .. code-block:: json

        {
          "A_to_B": {
            "strategy": "bsb.connectivity.VoxelIntersection",
            "presynaptic": {
              "cell_types": [
                "A"
              ],
            },
            "postsynaptic": {
              "cell_types": [
                "B"
              ],
            },
            "affinity": 0.5,
            "contacts": 1
          }
        }

    .. code-block:: python

      config.connectivity.add(
        "A_to_B",
         strategy="bsb.connectivity.VoxelIntersection",
         presynaptic=dict(cell_types=["A"]),
         postsynaptic=dict(cell_types=["type_B"]),
         affinity= 0.5,
         contacts= 1
      )

The previous example demonstrates a strategy to connect cells of type A with cells of type B,
where only half of the computed overlaps are considered, and one synapse is placed for each connection.
It is also possible to define the number of synapse per connection with a distribution:

.. tab-set-code::

    .. code-block:: json

            {
          "A_to_B": {
            "strategy": "bsb.connectivity.VoxelIntersection",
            "presynaptic": {
              "cell_types": [
                "A"
              ],
            },
            "postsynaptic": {
              "cell_types": [
                "B"
              ],
            },
            "affinity": 0.5,
            "contacts": {
              "distribution": "norm",
              "loc": 10,
              "scale": 2
            }
          }
        }

    .. code-block:: python

       config.connectivity.add(
         "A_to_B",
         strategy="bsb.connectivity.VoxelIntersection",
         presynaptic=dict(cell_types=["A"]),
         postsynaptic=dict(cell_types=["type_B"]),
         affinity= 0.5,
         contacts= dict(
           distribution="norm",loc=10,scale=2
         )
       )

In this case, the number of synapses is randomly drawn from a normal distribution
with a mean of 10 and a standard deviation of 2.

.. note::
  Normal distribution is just one option but all the distributions available in your scipy package
  can be used.