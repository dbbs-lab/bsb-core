.. _point_cloud:

###################################
Point cloud connectivity strategies
###################################

If the morphology of a cell is not at our disposal or if it is incomplete, a possibility is to use a point cloud instead, see B1 in :ref:`Bibliography`.
The morphology is approximated by a collection of geometric shapes representing the pre/postsynaptic neurites. Three different connectivity strategies are avaible.

MorphologyToCloudIntersection
*****************************

:class:`ShapesComposition  <.connectivity.point_cloud.geometric_shapes.ShapesComposition>` 
:class:`MorphologyToCloudIntersection <.connectivity.point_cloud.morphology_cloud_intersection.MorphologyToCloudIntersection>`

Create connections between the points of the morphology of the presynaptic cell and a point cloud representing a postsynaptic cell, checking if the points of the morphology are inside the geometric shapes representing the postsynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the presynaptic cell, but not of the postsynaptic cell.

CloudToMorphologyIntersection
*****************************

Class*: :class:`bsb.connectivity.CloudToMorphologyIntersection   <.connectivity.point_cloud.CloudToMorphologyIntersection>` 

Create connections between the point cloud representing the presynaptic cell the points of the morphology of a postsynaptic cell, checking if the points of the morphology are inside the geometric shapes representing the presynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the postsynaptic cell, but not of the presynaptic cell.

CloudToCloudIntersection
************************

Class*: :class:`bsb.connectivity.CloudToCloudIntersection   <.connectivity.point_cloud.CloudToCloudIntersection>` 

Create connections between the point cloud representing the presynaptic and postsynaptic cells. This strategy forms a connections generating a number of points inside the presynaptic probability cloud and checking if they are inside the geometric shapes representing the postsynaptic cell. One point per voxel is generated.
This connection strategy is suitable when we do not have a detailed morphology of neither the presynaptic nor the postsynaptic cell.

ShapesComposition
*****************

The geometric shapes representing a cell are stored in a :class:`ShapesComposition  <.connectivity.point_cloud.geometric_shapes.ShapesComposition>` object.
This class contains a collection of geometric shapes (objects of class :class:`GeometricShape  <.connectivity.point_cloud.geometric_shapes.GeometricShape>` ) along with a set of labels describing wich parts of a neuron each shape is describing. The parameter voxel_size controls how many points are generated in a cloud: it generates as many points as the number of voxels in the volume of the shapes.
The ShapesComposition object to use in a point cloud connection strategy must be specified in the `cloud_names` variable in the configuration file, specifying a list of files from which the point cloud are loaded. For example if we stored a ShapesComposition representing a stellate cell in a file called stellate.pck, we can connect a stellate cell to a Purkinje cell as follows.

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

The parameter `morphology_labels` specifies which shapes of the collection in ShapesComposition must be used, exactly as for the :class:`bsb.connectivity.VoxelIntersection` strategy.
The `affinity` parameter, as in :class:`bsb.connectivity.VoxelIntersection` strategy, controls the probability to form a connection.

Geometric shapes
----------------

There are four pre-defined GeometricShape already implemented in BSB: Ellipsoid, Cylinder, Cone and Sphere.
If needed, a user can define its own geometric shape, creating a new class inheriting from the base virtual class :class:`bsb.connectivity.GeometricShape  <.connectivity.point_cloud.GeometricShape>`.

Creating a ShapesComposition
----------------------------

We create a :class:`ShapesComposition  <.connectivity.point_cloud.geometric_shapes.ShapesComposition>` adding :class:`GeometricShape  <.connectivity.point_cloud.geometric_shapes.Geome>`  objects to a ShapesComposition object using the :meth:`bsb.connectivity.ShapesComposition.add_shape  <.connectivity.point_cloud.ShapesComposition.add_shape>`.
In the following example we represent the soma of a cell with a sphere, the axon with a cylinder and the dendritic tree with a cone. Then we save and plot the result, using :meth:`~.connectivity.point_cloud.geometric_shapes.ShapesComposition.save_to_file` and :meth:`~.plotting.plot_shape_wireframe`.

.. literalinclude:: create_shapescomposition.py
    :language: python

A `ShapesComposition` can be read from a file using the :meth:`bsb.connectivity.ShapesComposition.load_from_file  <.connectivity.point_cloud.ShapesComposition.load_from_file>`.


*Class*: :class:`bsb.connectivity.MorphologyToCloudIntersection   <.connectivity.point_cloud.MorphologyToCloudIntersection>` 

.. _Bibliography:

Bibliography
*****************

* B1: Gandolfi D, Mapelli J, Solinas S, De Schepper R, Geminiani A, Casellato C, D'Angelo E, Migliore M. A realistic morpho-anatomical connection strategy for modelling full-scale point-neuron microcircuits. Sci Rep. 2022 Aug 16;12(1):13864. doi: 10.1038/s41598-022-18024-y. Erratum in: Sci Rep. 2022 Nov 17;12(1):19792. PMID: 35974119; PMCID: PMC9381785.