.. _point_cloud:

###################################
Point cloud connectivity strategies
###################################

To reconstruct with great details the connections between 2 neurons, one needs to provide the
morphologies of these neurons. However, this data might be lacking or incomplete.
Moreover, the reconstruction of a detailed connectivity is computationally expensive as the program
have to find all apposition of the neurons arborizations.

To resolve these two issues, neurons can be represented as point clouds (see B1 in :ref:`Bibliography`).
Their morphology is here approximated by a collection of geometric shapes representing the
pre/postsynaptic neurites.

Creating simplified morphologies
********************************

The :class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition` allows the simplified
representation of cell morphologies. This class leverages a list geometric shapes
(:class:`~bsb.connectivity.point_cloud.geometric_shapes.GeometricShape`) to represent ``sections``
the cell morphology. Similarly to morphologies, labels should be associated to each of these
``sections``. These labels will be used as reference during connectivity.

For each ``section`` of the simplified morphology, the class samples a set of 3D points that belong
to it. This cloud of points is used to detect connections between a source and target neuron.
The points are uniformly distributed in the ``GeometricShape``, decomposing it into 3D voxels.
The program generates as many points as the number of voxels in the volume of the shapes.

Geometric shapes
----------------

Pre-defined GeometricShape implemented can be found in the ``~bsb.connectivity.point_cloud`` package.
Each shape has its own set of parameters. We provide here an example of the configuration
for a sphere:

.. code-block:: json

    "sphere":
    {
        "type": "sphere",
        "radius": 40.0,
        "center": [0, 0, 0]
    }

If needed, a user can define its own geometric shape, creating a new class inheriting from the base
virtual class :class:`~bsb.connectivity.point_cloud.geometric_shapes.GeometricShape`.

ShapesComposition
-----------------
To instantiate a :class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition`, you need
to provide a list of ``shapes`` together with their ``labels``: a list of lists of strings.
``shapes`` and ``labels`` should have the same size. For each shape, multiple labels can be provided.
You can additionally control the number of points sampled for connectivity with the parameter
``voxel_size``. This parameter corresponds to the side length of one voxel used to decompose the
shape collection.
Here represent the cell as a single sphere for the soma, a cone for the dendrites and a cylinder
for the axon:

.. code-block:: json

    "my_neuron":
    {
        "voxel_size": 25,
        "shapes":
        [
            {
                "type": "sphere",
                "radius": 40.0,
                "center": [0, 0, 0]},
            {
                "type": "cone",
                "center": [0, 0, 0],
                "radius": 100.0,
                "apex": [0, 100, 0]},
            {
                "type": "cylinder",
                "radius": 100.0,
                "top_center": [0, 0, 0],
                "bottom_center": [0, 0, 10]
            }
        ],
        "labels":
        [
            ["soma"],
            ["basal_dendrites", "apical_dendrites"],
            ["axon"]
        ],
    }

Point cloud connectivity
************************

The configuration of the point cloud strategies are similar to the other connectivity strategies (
see :class:`~bsb.connectivity.detailed.voxel_intersection.VoxelIntersection`).

The ``ShapesComposition`` configuration should be provided with the field ``shape_compositions`` in
the pre- and/or postsynaptic field (dependant on the strategy chosen).

The parameters ``morphology_labels`` here specifies which shapes of the ``shape_compositions`` in
:class:`~bsb.connectivity.point_cloud.geometric_shapes.ShapesComposition` must be used
(corresponds to values stored in ``labels``).

The ``affinity`` parameter controls the probability to form a connection.
Three different connectivity strategies based on ``ShapesComposition`` are available.

MorphologyToCloudIntersection
-----------------------------

The class :class:`~bsb.connectivity.point_cloud.morphology_cloud_intersection.MorphologyToCloudIntersection`
creates connections between the points of the morphology of the presynaptic cell and a point cloud
representing a postsynaptic cell, checking if the points of the morphology are inside the geometric
shapes representing the postsynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the presynaptic cell, but
not of the postsynaptic cell.

Configuration example:

.. code-block:: json

  "stellate_to_purkinje":
  {
    "strategy": "bsb.connectivity.MorphologyToCloudIntersection",
    "presynaptic": {
      "cell_types": ["stellate_cell"],
      "morphology_labels": ["axon"],
    },
    "postsynaptic": {
      "cell_types": ["purkinje_cell"],
      "morphology_labels": ["sc_targets"],
      "shape_compositions" : [{
        "voxel_size": 25,
        "shapes": [{"type": "sphere", "radius": 40.0, "center": [0, 0, 0]}],
        "labels": [["soma", "dendrites", "sc_targets", "axon"]],
      }]
    },
    "affinity": 0.1
  }

CloudToMorphologyIntersection
-----------------------------

The class :class:`~bsb.connectivity.point_cloud.cloud_morphology_intersection.CloudToMorphologyIntersection` creates connections between the point cloud representing the presynaptic cell the points of the morphology of a postsynaptic cell, checking if the points of the morphology are inside the geometric shapes representing the presynaptic cells.
This connection strategy is suitable when we have a detailed morphology of the postsynaptic cell,
but not of the presynaptic cell.

Configuration example:

.. code-block:: json

  "stellate_to_purkinje":
  {
    "strategy": "bsb.connectivity.CloudToMorphologyIntersection",
    "presynaptic": {
      "cell_types": ["stellate_cell"],
      "morphology_labels": ["axon"],
      "shape_compositions" : [{
        "voxel_size": 25,
        "shapes": [{"type": "sphere", "radius": 40.0, "center": [0, 0, 0]}],
        "labels": [["soma", "dendrites", "axon"]],
      }]
    },
    "postsynaptic": {
      "cell_types": ["purkinje_cell"],
      "morphology_labels": ["sc_targets"]
    },
    "affinity": 0.1
  }

CloudToCloudIntersection
------------------------

The class :class:`~bsb.connectivity.point_cloud.cloud_cloud_intersection.CloudToCloudIntersection`
creates connections between the point cloud representing the presynaptic and postsynaptic cells.
This strategy forms a connections generating a number of points inside the presynaptic probability
cloud and checking if they are inside the geometric shapes representing the postsynaptic cell.
One point per voxel is generated.
This connection strategy is suitable when we do not have a detailed morphology of neither the
presynaptic nor the postsynaptic cell.

Configuration example:

.. code-block:: json

  "stellate_to_purkinje":
  {
    "strategy": "bsb.connectivity.CloudToCloudIntersection",
    "presynaptic": {
      "cell_types": ["stellate_cell"],
      "morphology_labels": ["axon"],
      "shape_compositions" : [{
        "voxel_size": 25,
        "shapes": [{"type": "sphere", "radius": 40.0, "center": [0, 0, 0]}],
        "labels": [["soma", "dendrites", "axon"]],
      }]
    },
    "postsynaptic": {
      "cell_types": ["purkinje_cell"],
      "morphology_labels": ["sc_targets"],
      "shape_compositions" : [{
        "voxel_size": 25,
        "shapes": [{"type": "sphere", "radius": 40.0, "center": [0, 0, 0]}],
        "labels": [["soma", "dendrites", "sc_targets", "axon"]],
      }]
    },
    "affinity": 0.1
  }

.. _Bibliography:

Bibliography
************

* B1: Gandolfi D, Mapelli J, Solinas S, De Schepper R, Geminiani A, Casellato C, D'Angelo E, Migliore M. A realistic morpho-anatomical connection strategy for modelling full-scale point-neuron microcircuits. Sci Rep. 2022 Aug 16;12(1):13864. doi: 10.1038/s41598-022-18024-y. Erratum in: Sci Rep. 2022 Nov 17;12(1):19792. PMID: 35974119; PMCID: PMC9381785.