.. _point_cloud:

###################################
Point cloud connectivity strategies
###################################

If the morphology of a cell is not at our disposal or if it is incomplete, a possibility is to use a
point cloud instead, see B1 in :ref:`Bibliography`.
The morphology is approximated by a collection of geometric shapes representing the pre/postsynaptic
neurites. Three different connectivity strategies are available.

MorphologyToCloudIntersection
*****************************

The class :class:`~bsb.connectivity.point_cloud.morphology_cloud_intersection.MorphologyToCloudIntersection`
creates connections between the points of the morphology of the presynaptic cell and a point cloud
representing a postsynaptic cell, checking if the points of the morphology are inside the geometric
shapes representing the postsynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the presynaptic cell, but
not of the postsynaptic cell.

CloudToMorphologyIntersection
*****************************

The class :class:`~bsb.connectivity.point_cloud.cloud_morphology_intersection.CloudToMorphologyIntersection` creates connections between the point cloud representing the presynaptic cell the points of the morphology of a postsynaptic cell, checking if the points of the morphology are inside the geometric shapes representing the presynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the postsynaptic cell,
but not of the presynaptic cell.

CloudToCloudIntersection
************************

The class :class:`~bsb.connectivity.point_cloud.cloud_cloud_intersection.CloudToCloudIntersection`
creates connections between the point cloud representing the presynaptic and postsynaptic cells.
This strategy forms a connections generating a number of points inside the presynaptic probability
cloud and checking if they are inside the geometric shapes representing the postsynaptic cell.
One point per voxel is generated.
This connection strategy is suitable when we do not have a detailed morphology of neither the
presynaptic nor the postsynaptic cell.

ShapesComposition
*****************

The geometric shapes representing a cell are stored in a
:class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition` object.
This class contains a collection of geometric shapes (objects of class
:class:`~bsb.connectivity.point_cloud.geometric_shapes.GeometricShape` ) along with a set of labels
describing which parts of a neuron each shape is describing. The parameter voxel_size controls how
many points are generated in a cloud: it generates as many points as the number of voxels in the
volume of the shapes.
The ShapesComposition object to use in a point cloud connection strategy must be specified in the
``cloud_names`` variable in the configuration file, specifying a list of files from which the point
cloud are loaded. For example if we stored a ShapesComposition representing a stellate cell in a
file called stellate.pck, we can connect a stellate cell to a Purkinje cell as follows.

.. code-block:: json

  "stellate_to_purkinje": 
  {
    "strategy": "bsb.connectivity.CloudToMorphologyIntersection",
    "presynaptic": {
      "cell_types": ["stellate_cell"],
      "morphology_labels": ["axon"],
      "cloud_names" : ["stellate.pck"]
    },
    "postsynaptic": {
      "cell_types": ["purkinje_cell"],
      "morphology_labels": ["sc_targets"]
    },
    "affinity": 0.1
  }

The parameter ``morphology_labels`` specifies which shapes of the collection in
:class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition` must be used, exactly as
for the :class:`~bsb.connectivity.detailed.voxel_intersection.VoxelIntersection` strategy.
The ``affinity`` parameter, as in :class:`~bsb.connectivity.detailed.voxel_intersection.VoxelIntersection`
strategy, controls the probability to form a connection.

Geometric shapes
----------------

Pre-defined GeometricShape implemented can be found in the ``~bsb.connectivity.point_cloud`` package.
If needed, a user can define its own geometric shape, creating a new class inheriting from the base
virtual class :class:`~bsb.connectivity.point_cloud.geometric_shapes.GeometricShape`.

Creating a ShapesComposition
----------------------------

We create a :class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition` adding
:class:`~bsb.connectivity.point_cloud.geometric_shapes.GeometricShape`  objects to a
``ShapesComposition`` object using the
:meth:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition.add_shape`.
In the following example we represent the soma of a cell with a sphere, the axon with a cylinder and
the dendritic tree with a cone.
Then we save and plot the result, using
:meth:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition.save_to_file` and
``plotting.plot_shape_wireframe``.

.. literalinclude:: create_shapescomposition.py
    :language: python

A ``ShapesComposition`` can be read from a file using the method
:meth:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition.load_from_file`.

.. _Bibliography:

Bibliography
************

* B1: Gandolfi D, Mapelli J, Solinas S, De Schepper R, Geminiani A, Casellato C, D'Angelo E, Migliore M. A realistic morpho-anatomical connection strategy for modelling full-scale point-neuron microcircuits. Sci Rep. 2022 Aug 16;12(1):13864. doi: 10.1038/s41598-022-18024-y. Erratum in: Sci Rep. 2022 Nov 17;12(1):19792. PMID: 35974119; PMCID: PMC9381785.