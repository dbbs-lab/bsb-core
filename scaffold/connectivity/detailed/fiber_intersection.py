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
        FiberIntersection connection strategies voxelize a fiber and find its intersections with postsynaptic cells.
        It's a specific case of VoxelIntersection.

        For each presynaptic cell, the following steps are executed:
            1) extracting the FiberMorphology
            2) interpolate
            3) transform
            4) interpolate
            5) voxelize (generates the voxel_tree associated to this morphology)
            6) check intersections of presyn bounding box with all postsyn boxes
            7) check intersections of each candidate postsyn with current presyn voxel_tree

    """

    casts = {
        "convergence": int,
        "divergence": int,
        "affinity": float,
        "resolution": float,
    }

    defaults = {"affinity": 1.0, "resolution": 20.0}

    def validate(self):
        pass

    def connect(self):
        scaffold = self.scaffold

        p = index.Property(dimension=3)
        to_cell_tree = index.Index(properties=p)

        # Select all the cells from the pre- & postsynaptic type for a specific connection.
        from_type = self.from_cell_types[0]
        from_compartments = self.from_cell_compartments[0]
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

        for c, (from_cell, from_morpho) in enumerate(from_morphology_set):
            # Extract the FiberMorpho object for each branch in the from_compartments of the presynaptic morphology (1)
            compartments = from_morpho.get_compartments(
                compartment_types=from_compartments
            )
            morpho_rotation = from_cell.rotation
            fm = FiberMorphology(compartments, morpho_rotation)
            print("lenght comp fi ", len(compartments), len(fm.root_branches))
            print(fm.root_branches[0]._root._child)

            # Interpolate all branches recursively (2)
            self.interpolate_branches(fm.root_branches)

            # Transform (3). It requires the from_cell position that will be used for example in QuiverTransform to get
            # the orientation value in the voxel where the cell is located, while still keeping the morphology in its local reference frame.
            if self.transformation is not None:
                self.transformation.transform_branches(
                    fm.root_branches, from_cell.position
                )

            # Interpolate again (4)
            self.interpolate_branches(fm.root_branches)
            #

            # Voxelize all branches of the current morphology (5)
            from_bounding_box, from_voxel_tree, from_map, v_all = self.voxelize_branches(
                fm.root_branches, from_cell.position
            )

            # Check for intersections of the postsyn tree with the bounding box (6)

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
                # Same as in VoxelIntersection, only select a fraction of the total possible matches, based on how much
                # affinity there is between the cell types.
                # Affinity 1: All cells whose voxels intersect are considered to grow
                # towards eachother and always form a connection with other cells in their
                # voxelspace
                # Affinity 0: Cells completely ignore other cells in their voxelspace and
                # don't form connections.
                if np.random.rand() >= self.affinity:
                    continue
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
                        target_compartments.extend([from_map[from_voxel_id]])
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
                compartments_out.append([from_compartment.id, to_compartment])
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
            bounding_box = []
            # Initialize bottom and top extremes of bounding box to the start of the first branch compartment
            bounding_box.append(branches[0]._compartments[0].start + position)
            bounding_box.append(branches[0]._compartments[0].start + position)
        if voxel_tree is None:
            # Create Rtree
            from rtree import index

            p = index.Property(dimension=3)
            voxel_tree = index.Index(properties=p)
        if map is None:
            # Initialize map of compartment ids to empty list
            map = []

        v = 0

        for branch in branches:
            bounding_box, voxel_tree, map, v = branch.voxelize(
                position, bounding_box, voxel_tree, map
            )

            self.voxelize_branches(
                branch.child_branches, position, bounding_box, voxel_tree, map
            )

        return bounding_box, voxel_tree, map, v


class FiberTransform(ConfigurableClass):
    def transform_branches(self, branches, offset=None):
        # In QuiverTransform transform_branches, the offset is used to find the
        # orientation vector associated to the voxel where the compartment to
        # be rotated is located
        if offset is None:
            offset = np.zeros(3)
        for branch in branches:
            # @Robin: each branch has a start position in the reference frame of the morphology it belongs to or they all start at [0,0,0]?
            self.transform_branch(branch, offset)
            self.transform_branches(branch.child_branches, offset)

    @abc.abstractmethod
    def transform_branch(self):
        pass


class QuiverTransform(FiberTransform):
    """
        QuiverTransform applies transformation to a FiberMorphology, based on an orientation field in a voxelized volume.
        Used for parallel fibers.
    """

    # Class attributes

    casts = {"vol_res": float}

    defaults = {"vol_res": 1.0, "quivers": [1.0, 1.0, 1.0]}

    def validate(self):

        if self.shared is True:
            raise ConfigurationError(
                "Attribute 'shared' can't be True for {} transformation".format(self.name)
            )

    def transform_branch(self, branch, offset):

        """
            Compute bending transformation of a fiber branch (discretized according to original compartments and configured resolution value).
            The transformation is a rotation of each segment/compartment of each fiber branch to align to the cross product between
            the orientation vector and the transversal direction vector (i.e. cross product between fiber morphology/parent branch orientation
            and branch direction):
            compartment[n+1].start = compartment[n].end
            cross_prod = orientation_vector X transversal_vector or transversal_vector X orientation_vector
            compartment[n+1].end = compartment[n+1].start + cross_prod * length_comp

            :param branch: a branch of the current fiber to be transformed
            :type branch: Branch object
            :returns: a transformed branch

        """

        # Only QuiverTransform has the attribute quivers, giving the orientation in a discretized volume of size volume_res
        if self.quivers is not None:
            orientation_data = self.quivers
        else:
            raise AttributeError("Missing  attribute 'quivers' for {}".format(self.name))
        if hasattr(self, "vol_res"):
            volume_res = self.vol_res
        else:
            raise AttributeError("Missing  attribute 'vol_res' for {}".format(self.name))

        # Bypass for testing
        orientation_data = np.ones(shape=(3, 500, 500, 500))
        volume_res = 25

        # We really need to check if shared here? We are applying to each cell - so this should be checked before in the code
        if not self.shared:
            # Compute branch direction - to check that PFs have 2 branches, left and right
            branch_dir = branch._compartments[0].end - branch._compartments[0].start
            # Normalize branch_dir vector
            branch_dir = branch_dir / np.linalg.norm(branch_dir)

            num_comp = len(branch._compartments)

            # Looping over branch compartments to transform them
            for comp in range(len(branch._compartments)):
                # Find direction transversal to branch: cross product between
                # the branch direction and the original morphology/parent branch
                if branch.orientation is None:
                    transversal_vector = np.cross([0, 1, 0], branch_dir)
                else:
                    transversal_vector = np.cross(branch.orientation, branch_dir)

                # Extracting index of voxel where the current compartment is located
                voxel_ind = (branch._compartments[comp].start + offset) / volume_res

                voxel_ind = voxel_ind.astype(int)
                orientation_vector = orientation_data[
                    :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                ]
                cross_prod = np.cross(branch_dir, orientation_vector)
                cross_prod = cross_prod / np.linalg.norm(cross_prod)
                length_comp = np.linalg.norm(
                    branch._compartments[comp].end - branch._compartments[comp].start
                )
                # Transform compartment
                branch._compartments[comp].end = (
                    branch._compartments[comp].end + cross_prod * length_comp
                )
                if comp < (num_comp - 1):
                    # The new end is the start of the adjacent compartment
                    branch._compartments[comp + 1].start = branch._compartments[comp].end
