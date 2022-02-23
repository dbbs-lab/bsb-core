import numpy as np
import math
from ..strategy import ConnectionStrategy
from .shared import Intersectional
from ... import config
from ...config import types
from ...exceptions import *
from ...morphologies import Morphology, Branch
from ...plotting import plot_fiber_morphology
from ...reporting import report, warn
import abc

# Import rtree
from rtree import index
from rtree.index import Rtree


class FiberTransform(abc.ABC):
    def boot(self):
        self._branch_cut_num = 0

    def transform_branches(self, branches, offset=None):
        if offset is None:
            offset = np.zeros(3)
        for branch in branches:
            self.transform_branch(branch, offset)
            self.transform_branches(branch.child_branches, offset)

    @abc.abstractmethod
    def transform_branch(self):
        pass


@config.node
class FiberIntersection(Intersectional, ConnectionStrategy):
    """
    FiberIntersection connection strategies voxelize a fiber and find its intersections with postsynaptic cells.
    It's a specific case of VoxelIntersection.

    For each presynaptic cell, the following steps are executed:

    #. Extract the FiberMorphology
    #. Interpolate points on the fiber until the spatial resolution is respected
    #. transform
    #. Interpolate points on the fiber until the spatial resolution is respected
    #. Voxelize (generates the voxel_tree associated to this morphology)
    #. Check intersections of presyn bounding box with all postsyn boxes
    #. Check intersections of each candidate postsyn with current presyn voxel_tree

    """

    affinity = config.attr(default=1.0)
    contacts = config.attr(type=types.distribution(), default=1)
    resolution = config.attr(default=20.0)
    to_plot = config.attr(type=list)
    transformation = config.attr(type=FiberTransform)

    def connect(self):
        scaffold = self.scaffold

        p = index.Property(dimension=3)
        to_cell_tree = index.Index(properties=p)
        labels_pre = None if self.label_pre is None else [self.label_pre]
        labels_post = None if self.label_post is None else [self.label_post]

        # Select all the cells from the pre- & postsynaptic type for a specific connection.
        from_type = self.from_cell_types[0]
        from_compartments = self.from_cell_compartments[0]
        to_compartments = self.to_cell_compartments[0]
        to_type = self.to_cell_types[0]
        from_ps = self.scaffold.get_placement_set(from_type.name, labels=labels_pre)
        to_ps = self.scaffold.get_placement_set(to_type.name, labels=labels_post)

        # Load the morphology and voxelization data for the entrire morphology, for each cell type.
        from_morphology_set = from_placement_set.load_morphologies()

        to_morphology_set = to_placement_set.load_morphologies()
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

        connections_out = []
        compartments_out = []
        morphologies_out = []

        fig = None
        fiber_cut_num = 0
        for c, (from_cell, from_morpho) in enumerate(from_morphology_set):
            # (1) Extract the FiberMorpho object for each branch in the from_compartments
            # of the presynaptic morphology
            compartments = from_morpho.get_compartments(from_compartments)
            if not compartments:
                warn(f"Missing `{from_compartments}` compartments.")
                continue
            morpho_rotation = from_cell.rotation
            fm = FiberMorphology(compartments, morpho_rotation)

            # (2) Interpolate all branches recursively
            self.interpolate_branches(fm.root_branches)

            if c in self.to_plot:
                fig = plot_fiber_morphology(
                    fm, fig=fig, offset=from_cell.position, show=False
                )

            # (3) Transform the fiber if present.
            # It requires the from_cell position that will be
            # used for example in QuiverTransform to get the orientation value
            # in the voxel where the cell is located, while still keeping the
            # morphology in its local reference frame.
            if self.transformation is not None:
                self.transformation.transform_branches(
                    fm.root_branches, from_cell.position
                )
                if self.transformation._branch_cut_num > 0:
                    fiber_cut_num += 1

            if c in self.to_plot:
                fig = plot_fiber_morphology(fm, fig=fig, offset=from_cell.position)

            # (4) Interpolate again
            self.interpolate_branches(fm.root_branches)

            # (5) Voxelize all branches of the transformed fiber morphology
            p = index.Property(dimension=3)
            # The bounding box is incrementally expanded, these initial bounds are a point
            # at the start of the first root branch.
            from_bounding_box = [
                fm.root_branches[0]._compartments[0].start + from_cell.position
            ] * 2
            from_voxel_tree = index.Index(properties=p)
            from_map = []
            (
                from_bounding_box,
                from_voxel_tree,
                from_map,
                voxel_list,
            ) = self.voxelize_branches(
                fm.root_branches,
                from_cell.position,
                from_bounding_box,
                from_voxel_tree,
                from_map,
            )

            # (6) Check for intersections of the postsyn tree with the bounding box

            ## TODO: Check if bounding box intersection is convenient

            # Bounding box intersection to identify possible connected candidates, using
            # the bounding box of the point cloud. Query the Rtree for intersections of
            # to_cell boxes with our from_cell box
            cell_intersections = list(
                to_cell_tree.intersection(
                    tuple(np.concatenate(from_bounding_box)), objects=False
                )
            )

            # (7) For each hit on the box intersection between pre- and postsynaptic
            # cells, perform voxel cloud intersection to identify actually connected cell
            # pairs and select compartments from their intersecting voxels to form
            # connections with.
            for partner in cell_intersections:
                # Same as in VoxelIntersection, only select a fraction of the total
                # possible matches, based on how much affinity there is between the cell
                # types.
                if np.random.rand() >= self.affinity:
                    continue
                # Get the precise morphology of the to_cell we collided with
                to_cell, to_morpho = to_morphology_set[partner]
                # Get the map from voxel id to list of compartments in that voxel.
                to_map = to_morpho.cloud.map
                # Find which voxels inside the bounding box of the fiber and the cell box
                # actually intersect with eachother.
                voxel_intersections = self.intersect_voxel_tree(
                    from_voxel_tree, to_morpho.cloud, to_cell.position
                )
                # Returns a list of lists: the elements in the inner lists are the indices
                # of the voxels in the from point cloud, the indices of the lists inside
                # of the outer list are the to voxel indices.
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
                        target_compartments.extend([from_map[from_voxel_id]])
                    target_comps_per_to_voxel[to_voxel_id] = target_compartments
                # Weigh the random sampling by the amount of compartments so
                # that voxels with more compartments have a higher chance of
                # having one of their many compartments randomly picked.
                voxel_weights = [
                    len(to_map[to_voxel_id]) * len(from_targets)
                    for to_voxel_id, from_targets in target_comps_per_to_voxel.items()
                ]
                weight_sum = sum(voxel_weights)
                voxel_weights = [w / weight_sum for w in voxel_weights]
                contacts = round(self.contacts.sample())
                # Pick a random voxel and its targets
                candidates = list(target_comps_per_to_voxel.items())
                while contacts > 0:
                    contacts -= 1
                    # Pick a random voxel and its targets
                    random_candidate_id = np.random.choice(
                        range(len(candidates)), 1, p=voxel_weights
                    )[0]
                    # Pick a to_voxel_id and its target compartments from the list of candidates
                    random_to_voxel_id, random_compartments = candidates[
                        random_candidate_id
                    ]
                    # Pick a random from and to compartment of the chosen voxel pair
                    from_compartment = np.random.choice(random_compartments, 1)[0]
                    to_compartment = np.random.choice(to_map[random_to_voxel_id], 1)[0]
                    compartments_out.append([from_compartment.id, to_compartment])
                    connections_out.append([from_cell.id, to_cell.id])

        # Throw warning on cut fibers:
        if fiber_cut_num > 0:
            warn(
                "{} fibers out of {} were cut due to outside of quiver volume or external region voxels.".format(
                    fiber_cut_num, c + 1
                ),
                QuiverFieldWarning,
            )
        self.scaffold.connect_cells(
            self,
            np.array(connections_out or np.empty((0, 2))),
            compartments=np.array(compartments_out or np.empty((0, 2))),
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
        self,
        branches,
        position,
        bounding_box=None,
        voxel_tree=None,
        map=None,
        voxel_list=None,
    ):
        voxel_list = []
        for branch in branches:
            bounding_box, voxel_tree, map, voxel_list = branch.voxelize(
                position, bounding_box, voxel_tree, map, voxel_list
            )
            self.voxelize_branches(
                branch.child_branches, position, bounding_box, voxel_tree, map, voxel_list
            )

        return bounding_box, voxel_tree, map, voxel_list


class QuiverTransform(FiberTransform):
    """
    QuiverTransform applies transformation to a FiberMorphology, based on an orientation field in a voxelized volume.
    Used for parallel fibers.
    """

    # Class attributes

    casts = {"vol_res": float}

    defaults = {
        "vol_res": 10.0,
        "vol_start": [0.0, 0.0, 0.0],
        "quivers": None,
    }

    def validate(self):
        if self.shared is True:
            raise ConfigurationError(
                "Attribute 'shared' can't be True for {} transformation".format(self.name)
            )

        # Only QuiverTransform has the attribute quivers, giving the orientation in a
        # discretized volume of size volume_res
        if self.quivers is None:
            raise AttributeError("Missing  attribute 'quivers' for {}".format(self.name))

        if type(self.quivers) is not np.ndarray:
            self.quivers = np.array(self.quivers)

        if not hasattr(self, "vol_res"):
            raise AttributeError("Missing  attribute 'vol_res' for {}".format(self.name))

        if not hasattr(self, "vol_start"):
            raise AttributeError(
                "Missing  attribute 'vol_start' for {}".format(self.name)
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
        :type branch: :~class:`.morphologies.Branch`
        :returns: a transformed branch
        :rtype: :~class:`.morphologies.Branch`

        """
        orientation_data = self.quivers
        volume_res = self.vol_res
        volume_start = self.vol_start

        if not self.shared:
            # Compute branch direction - to check that PFs have 2 branches, left and right
            branch_dir = self.get_branch_direction(branch)
            # If the entire branch consists of compartments without direction, do nothing.
            if branch_dir is False:
                return

            num_comp = len(branch._compartments)

            # Compute length of the first compartment in the current branch
            length_comp = np.linalg.norm(
                branch._compartments[0].end - branch._compartments[0].start
            )
            # Looping over branch compartments to transform them
            for comp in range(len(branch._compartments)):
                # Find direction transversal to branch: cross product between
                # the branch direction and the original morphology/parent branch
                if branch.orientation is None:
                    transversal_vector = np.cross(branch_dir, [0, 1, 0])
                else:
                    transversal_vector = np.cross(branch_dir, branch.orientation)

                # Extracting index of voxel where the current compartment is located
                voxel_ind = (
                    branch._compartments[comp].start + offset - volume_start
                ) / volume_res

                voxel_ind = voxel_ind.astype(int) - [1, 1, 1]
                # Catch values falling outside of quiver field volume
                if (np.array(voxel_ind) < np.array([0, 0, 0])).all() or (
                    np.array(voxel_ind) > np.array(orientation_data.shape[1:])
                ).all():
                    # Update number of cut branches
                    self._branch_cut_num += 1
                    # Detach subsequent compartments from branch
                    leftover_branch = branch.detach(branch._compartments[comp])
                    break

                orientation_vector = orientation_data[
                    :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                ]

                # Catch values belonging to a different area than the reconstructed one (marked by NaN)
                if np.isnan(orientation_vector).any():
                    self._branch_cut_num += 1
                    # Detach subsequent compartments from branch
                    leftover_branch = branch.detach(branch._compartments[comp])
                    break

                cross_prod = np.cross(orientation_vector, transversal_vector)
                cross_prod = cross_prod / np.linalg.norm(cross_prod)

                # Transform compartment
                branch._compartments[comp].end = (
                    branch._compartments[comp].start + cross_prod * length_comp
                )
                if comp < (num_comp - 1):
                    length_comp = np.linalg.norm(
                        branch._compartments[comp + 1].end
                        - branch._compartments[comp + 1].start
                    )
                    # The new end is the start of the adjacent compartment
                    branch._compartments[comp + 1].start = branch._compartments[comp].end

    def get_branch_direction(self, branch):
        for comp in branch._compartments:
            branch_dir = comp.end - comp.start
            if not np.sum(branch_dir):
                continue
            # Normalize branch_dir vector
            branch_dir = branch_dir / np.linalg.norm(branch_dir)
            return branch_dir
