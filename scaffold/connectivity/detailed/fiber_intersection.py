import numpy as np
import math
from ..strategy import ConnectionStrategy
from .shared import MorphologyStrategy
from ...models import MorphologySet
from ...exceptions import *
from ...helpers import ConfigurableClass
from ...networks import FiberMorphology, Branch
import abc

# Import rtree & instantiate the index with its properties.
from rtree import index
from rtree.index import Rtree


class FiberIntersection(ConnectionStrategy, MorphologyStrategy):
    """
        Description
    """

    casts = {"convergence": int, "divergence": int}

    def validate(self):
        pass

    def connect(self):
        scaffold = self.scaffold

        p = index.Property(dimension=3)
        to_cell_tree = index.Index(properties=p)

        # Select all the cells from the pre- & postsynaptic type for a specific connection.
        from_type = self.from_cell_types[0]
        from_compartments = self.from_cell_compartments[0]
        print("from compartments ", from_compartments)
        to_compartments = self.to_cell_compartments[0]
        to_type = self.to_cell_types[0]
        from_placement_set = self.scaffold.get_placement_set(from_type.name)
        to_placement_set = self.scaffold.get_placement_set(to_type.name)
        from_cells = self.scaffold.get_cells_by_type(from_type.name)
        to_cells = self.scaffold.get_cells_by_type(to_type.name)

        # Load the morphology and voxelization data for the entrire morphology, for each cell type.
        from_morphology_set = MorphologySet(
            scaffold, from_type, from_placement_set, compartment_types=from_compartments
        )
        to_morphology_set = MorphologySet(
            scaffold, to_type, to_placement_set, compartment_types=to_compartments
        )
        joined_map = (
            from_morphology_set._morphology_map + to_morphology_set._morphology_map
        )
        joined_map_offset = len(from_morphology_set._morphology_map)

        # REWORKING: 1 single loop doing everything for that postsyn cell:
        #    1) extracting the FiberMorpho
        #    2) interpolate
        #    3) transform
        #    4) interpolate
        #    5) voxelize (generates the voxel_tree associated to this morphology)
        #    6) check intersections of presyn bounding box with all postsyn boxes
        #    7) check intersections of each candidate postsyn with current presyn voxel_tree

        for c, (from_cell, from_morpho) in enumerate(from_morphology_set):
            # Extract the FiberMorpho object for each branch in the morphology (1)
            compartments = from_morpho.get_compartments()
            fm = FiberMorphology(compartments)
            # Interpolate all branches recursively (2)
            self.interpolate_branches(fm.root_branches)
            # Transform (3)
            if self.transformation is not None:
                self.transformation.transform_branches(
                    fm.root_branches, from_cell.position
                )

            # Interpolate again (4)
            self.interpolate_branches(fm.root_branches)
            #

            # Voxelize all branches of the current morphology (5)
            from_bounding_box, from_voxel_tree, from_map = self.voxelize_branches(
                fm.root_branches, from_cell.position
            )

            # Check for intersections of the postsyn tree with the bounding box (6)
            # For every postsynaptic cell, derive the box incorporating all voxels,
            # and store that box in the tree, to later find intersections with that cell.
            for i, (to_cell, morphology) in enumerate(to_morphology_set):
                self.assert_voxelization(morphology, to_compartments)
                to_offset = np.concatenate((to_cell.position, to_cell.position))
                to_box = morphology.cloud.get_voxel_box()
                to_cell_tree.insert(i, tuple(to_box + to_offset))

            ## TODO: Check if bounding box intersection is convenient

            # Bounding box intersection to identify possible connected candidates, using the bounding box of the point cloud
            # Query the Rtree for intersections of to_cell boxes with our from_cell box
            cell_intersections = list(
                to_cell_tree.intersection(
                    tuple(np.concatenate(from_bounding_box)), objects=False
                )
            )

            # For candidate postsyn intersecting cells, find the intersecting from voxels/compartments (7)
            # Voxel cloud intersection to identify real connected cells and their compartments
            # Loop over each intersected partner to find and select compartment intersections
            for partner in cell_intersections:
                # Get the precise morphology of the to_cell we collided with
                to_cell, to_morpho = to_morphology_set[partner]
                # Get the map from voxel id to list of compartments in that voxel.
                to_map = to_morpho.cloud.map
                # Find which voxels inside the bounding box of the fiber and the cell box actually intersect with eachother.
                voxel_intersections = self.intersect_voxel_tree(
                    from_voxel_tree, to_morpho.cloud, to_cell.position
                )
                # Returns a list of lists: the elements in the inner lists are the indices of the
                # voxels in the from point cloud, the indices of the lists inside of the outer list
                # are the to voxel indices.
                #
                # Find non-empty lists: these voxels actually have intersections
                intersecting_to_voxels = np.nonzero(voxel_intersections)[0]
                if not len(intersecting_to_voxels):
                    # No intersections found? Do nothing, continue to next partner.
                    continue
                # Dictionary that stores the target compartments for each to_voxel.
                target_comps_per_to_voxel = {}
                # Iterate over each to_voxel index.
                for to_voxel_id in intersecting_to_voxels:
                    # Get the list of voxels that the to_voxel intersects with.
                    intersecting_voxels = voxel_intersections[to_voxel_id]
                    target_compartments = []
                    for from_voxel_id in intersecting_voxels:
                        # Store all of the compartments in the from_voxel as
                        # possible candidates for these cells' connections
                        # @Robin: map should contain comp.id or comp???
                        target_compartments.extend(from_map[from_voxel_id])
                    target_comps_per_to_voxel[to_voxel_id] = target_compartments
                # Weigh the random sampling by the amount of compartments so that voxels
                # with more compartments have a higher chance of having one of their many
                # compartments randomly picked.
                voxel_weights = [
                    len(to_map[to_voxel_id]) * len(from_targets)
                    for to_voxel_id, from_targets in target_comps_per_to_voxel.items()
                ]
                weight_sum = sum(voxel_weights)
                voxel_weights = [w / weight_sum for w in voxel_weights]
                # Pick a random voxel and its targets
                candidates = list(target_comps_per_to_voxel.items())
                random_candidate_id = np.random.choice(
                    range(len(candidates)), 1, p=voxel_weights
                )[0]
                # Pick a to_voxel_id and its target compartments from the list of candidates
                random_to_voxel_id, random_compartments = candidates[random_candidate_id]
                # Pick a random from and to compartment of the chosen voxel pair
                from_compartment = np.random.choice(random_compartments, 1)[0]
                to_compartment = np.random.choice(to_map[random_to_voxel_id], 1)[0]
                compartments_out.append([from_compartment, to_compartment])
                morphologies_out.append(
                    [from_morpho._set_index, joined_map_offset + to_morpho._set_index]
                )
                connections_out.append([from_cell.id, to_cell.id])

        self.scaffold.connect_cells(
            self,
            np.array(connections_out or np.empty((0, 2))),
            morphologies=np.array(morphologies_out or np.empty((0, 2), dtype=str)),
            compartments=np.array(compartments_out or np.empty((0, 2))),
            morpho_map=joined_map,
        )

        # For every presynaptic cell, build a list of from_points in it as a point collection
        # The variable from_points will be a 2D list of num_presyn_cell x point_per
        from_points = []
        for i, (from_cell, from_morpho) in enumerate(from_morphology_set):
            from_points.append([])
            for c in from_morpho.compartments:
                from_points[i].append(from_cell.position + c.start)
                ## TODO: add branching topology to separate e.g. left and right fibers
                # Check for resolution and interpolate between start and end of compartments to match the resolution
                length_comp = np.linalg.norm(c.end - c.start)

                if length_comp > self.resolution:
                    ## TODO: add replication of end and start points for each new segment
                    ## TODO: take into account possible fibers parallel to main axes leading to division by 0
                    num_to_add = math.ceil(length_comp / self.resolution)
                    x_to_add = list(np.linspace(c.start[0], c.end[0], num_to_add))
                    y_to_add = list(
                        c.start[1]
                        + ((x_to_add - c.start[0]) / (c.end[0] - c.start[0]))
                        * (c.end[1] - c.start[1])
                    )
                    z_to_add = list(
                        c.start[2]
                        + ((x_to_add - c.start[0]) / (c.end[0] - c.start[0]))
                        * (c.end[2] - c.start[2])
                    )
                    from_points[i].extend(
                        from_cell.position
                        + list(map(list, zip(x_to_add, y_to_add, z_to_add)))
                    )

                from_points[i].append(from_cell.position + c.end)

        if self.transformation is not None:
            from_points = self.transformation.transform(from_points)

        # Re-do the interpolation, in case the transformation has changed segment lengths (add a function for it)

        # For every postsynaptic cell, derive the box incorporating all voxels,
        # and store that box in the tree, to later find intersections with that cell.
        for i, (to_cell, morphology) in enumerate(to_morphology_set):
            self.assert_voxelization(morphology, to_compartments)
            to_offset = np.concatenate((to_cell.position, to_cell.position))
            to_box = morphology.cloud.get_voxel_box()
            to_cell_tree.insert(i, tuple(to_box + to_offset))

        # For each presynaptic cell, find all postsynaptic cells that its outer
        # box intersects with.
        connections_out = []
        compartments_out = []
        morphologies_out = []
        for c, (from_cell, from_morphology) in enumerate(
            from_morphology_set
        ):  # maybe better enumerate

            current_from_points_array = np.array(from_points[c])
            # Finding bounding box for the actual fiber point cloud
            bounding_box = tuple(
                np.concatenate(
                    (
                        np.amin(current_from_points_array, axis=0),
                        np.amax(current_from_points_array, axis=0),
                    )
                )
            )
            ## TODO: Check if bounding box intersection is convenient

            # Bounding box intersection to identify possible connected candidates, using the bounding box of the point cloud
            # Query the Rtree for intersections of to_cell boxes with our from_cell box
            cell_intersections = list(
                to_cell_tree.intersection(bounding_box, objects=False)
            )

    def intersect_voxel_tree(self, from_voxel_tree, to_cloud, to_pos):
        """
            Similarly to `intersect_clouds` from `VoxelIntersection`, it finds intersecting voxels between a from_voxel_tree
            and a to_cloud set of voxels

            :param from_voxel_tree: tree built from the voxelization of all branches in the fiber (in absolute coordinates)
            :type from_point_cloud: Rtree index
            :param to_cloud: voxel cloud associated to a to_cell morphology
            :type to_cloud: `VoxelCloud`
            :param to_pos: 3-D position of to_cell neuron
            :type to_pos: list
        """

        voxel_intersections = []

        # Find intersection of to_cloud with from_voxel_tree
        for v, voxel in enumerate(to_cloud.get_voxels(cache=True)):
            absolute_position = np.add(voxel, to_pos)
            absolute_box = np.add(absolute_position, to_cloud.grid_size)
            box = np.concatenate((absolute_position, absolute_box))
            voxel_intersections.append(
                list(from_voxel_tree.intersection(tuple(box), objects=False))
            )
        return voxel_intersections

    def assert_voxelization(self, morphology, compartment_types):
        if len(morphology.cloud.get_voxels()) == 0:
            raise IncompleteMorphologyError(
                "Can't intersect without any {} in the {} morphology".format(
                    ", ".join(compartment_types), morphology.morphology_name
                )
            )

    def interpolate_branches(self, branches):
        for branch in branches:
            branch.interpolate(self.resolution)
            self.interpolate_branches(branch.child_branches)

    def voxelize_branches(
        self, branches, position, bounding_box=None, voxel_tree=None, map=None
    ):
        if bounding_box is None:
            bounding = []
            # Initialize bottom and top extremes of bounding box to the start of the first branch compartment
            bounding_box.append(branches[0]._compartments[0].start + position)
            bounding_box.append(branches[0]._compartments[0].start + position)
        if tree is None:
            # Create Rtree
            from rtree import index

            p = index.Property(dimension=3)
            voxel_tree = index.Index(properties=p)
        if map is None:
            # Initialize map of compartment ids to empty list
            map = []

        for branch in branches:
            bounding_box, voxel_tree, map = branch.voxelize(
                position, bounding_box, voxel_tree, map
            )
            self.voxelize_branches(
                branch.child_branches, position, bounding_box, voxel_tree, map
            )
        return bounding_box, voxel_tree, map


class FiberTransform(ConfigurableClass):
    @abc.abstractmethod
    def transform(self):
        pass


class QuiverTransform(FiberTransform):
    """
        QuiverTransform applies transformation to a Morphology, based on an orientation field in a voxelized volume.
        Used for parallel fibers.
    """

    def validate(self):

        if self.shared is True:
            raise ConfigurationError(
                "Attribute 'shared' can't be True for {} transformation".format(self.name)
            )

    def transform(self, fiber_morpho):
        # transform(self, fiber_morpho)interpolate
        """
            Compute bending transformation of a point cloud representing the discretization of a fiber (according to
            original compartments and configured resolution value).
            The transformation is a rotation of each segment/compartment (identified by a point_start and point_end) of the fiber
            to align to the cross product between the orientation vector and the transversal direction vector:
            new_point_start = old_point_start
            cross_prod = orientation_vector X transversal_vector
            new_point_end = point_start + cross_prod * length_comp

            The function is used for bifurcated fibers, bending the left and right branches according to the left and right
            transversal vectors.

            :param point_cloud: a set of from_points representing segments of each fiber in the placement_set to be connected
            :type point_cloud: 2-D list
            :returns: a transformed point could (2-D list)

        """
        # Left and right transversal vectors
        trans_vector_lx = [0, 0, -1]
        trans_vector_rx = [0, 0, 1]

        # Only QuiverTransform has the attribute quivers, giving the orientation in a discretized volume of size volume_res
        if self.quivers is not None:
            orientation_data = self.quivers
        else:
            raise AttributeError("Missing  attribute 'quivers' for {}".format(self.name))
        if self.vol_res is not None:
            volume_res = self.vol_res
        else:
            raise AttributeError("Missing  attribute 'vol_res' for {}".format(self.name))

        # Bypass for testing
        orientation_data = np.ones(shape=(3, 500, 500, 500))
        volume_res = 1

        if not self.shared:
            # Loop over all cells
            for cell in range(len(point_cloud)):
                # First 4 elements are the first compartment from_points (start and end) of each initial compartment of the 2 (parallel fiber) branches
                # Therefore, the loop moves in steps of 4
                for comp in range(0, len(point_cloud[cell]), 4):
                    # Left branch - first 2 elements
                    voxel_ind = point_cloud[cell][comp] / volume_res
                    voxel_ind = voxel_ind.astype(int)
                    print(voxel_ind)
                    orientation_vector = orientation_data[
                        :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                    ]
                    cross_prod = np.cross(orientation_vector, trans_vector_lx)
                    cross_prod = cross_prod / np.linalg.norm(cross_prod)
                    length_comp = np.linalg.norm(
                        point_cloud[cell][comp + 1] - point_cloud[cell][comp]
                    )
                    point_cloud[cell][comp + 1] = (
                        point_cloud[cell][comp] + cross_prod * length_comp
                    )
                    # The new end is the nex start of the adjacent compartment
                    point_cloud[cell][comp + 4] = point_cloud[cell][comp + 1]

                    # Right branch
                    voxel_ind = point_cloud[cell][comp + 2] / volume_res
                    voxel_ind = voxel_ind.astype(int)
                    orientation_vector = orientation_data[
                        :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                    ]
                    point_cloud[cell][comp + 3] = point_cloud[cell][comp + 2] + np.cross(
                        orientation_vector, trans_vector_rx
                    )
                    # The new end is the nex start of the adjacent compartment
                    point_cloud[cell][comp + 6] = point_cloud[cell][comp + 3]

        return point_cloud

    def transform_branches(self, branches, offset=None):
        if offset is None:
            offset = np.zeros(3)
        for branch in branches:
            branch_offset = offset + transform_branch(branch, offset)
            self.transform_branches(branch.child_branches, branch_offset)

    def transform_branch(self, branch, offset):
        pass
