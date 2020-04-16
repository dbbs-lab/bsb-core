import numpy as np
from ..strategy import ConnectionStrategy
from .shared import MorphologyStrategy
from ...models import MorphologySet
from ...exceptions import *
from ...helpers import ConfigurableClass
import abc


class FiberIntersection(ConnectionStrategy, MorphologyStrategy):
    """
        Description
    """

    casts = {"convergence": int, "divergence": int}

    def validate(self):
        pass

    def connect(self):
        scaffold = self.scaffold

        # Import rtree & instantiate the index with its properties.
        from rtree import index
        from rtree.index import Rtree

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

        # For every presynaptic cell build a list of points in it as a point collection
        points = []
        for i, (from_cell, from_morpho) in enumerate(from_morphology_set):
            points.append([])
            for c in from_morpho.compartments:
                points[i].append(c.start)
                points[i].append(c.end)

        if self.quivers is not None:
            orientation = self.quivers

        points = self.transformation.transform(points, orientation)

        # # check for resolution
        # for p in points:
        #     # Query the Rtree for intersections of to_cell boxes with our from_cell box
        #     cell_intersections = list(to_cell_tree.intersection(this_box, objects=False))
        #
        #

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
        for from_cell, from_morpho in from_morphology_set:
            # Make sure that the voxelization was successful
            self.assert_voxelization(from_morpho, from_compartments)
            # Get the outer box of the morphology.
            from_box = from_morpho.cloud.get_voxel_box()
            # Get a map from voxel index to compartments in that voxel.
            from_map = from_morpho.cloud.map
            # Transform the box into a rectangle that we can query the Rtree with.
            this_box = tuple(
                from_box + np.concatenate((from_cell.position, from_cell.position))
            )
            # Query the Rtree for intersections of to_cell boxes with our from_cell box
            cell_intersections = list(to_cell_tree.intersection(this_box, objects=False))

            # Loop over each intersected partner to find and select compartment intersections
            for partner in cell_intersections:
                # Get the precise morphology of the to_cell we collided with
                to_cell, to_morpho = to_morphology_set[partner]
                # Get the map from voxel id to list of compartments in that voxel.
                to_map = to_morpho.cloud.map
                # Find which voxels inside the cell boxes actually intersect with eachother.
                voxel_intersections = self.intersect_clouds(
                    from_morpho.cloud,
                    to_morpho.cloud,
                    from_cell.position,
                    to_cell.position,
                )
                # Returns a list of lists: the elements in the inner lists are the indices of the
                # voxels in the from morphology, the indices of the lists inside of the outer list
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

    def intersect_clouds(self, from_cloud, to_cloud, from_pos, to_pos):
        voxel_intersections = []
        translation = to_pos - from_pos
        for v, voxel in enumerate(to_cloud.get_voxels(cache=True)):
            relative_position = np.add(voxel, translation)
            relative_box = np.add(relative_position, to_cloud.grid_size)
            box = np.concatenate((relative_position, relative_box))
            voxel_intersections.append(
                list(from_cloud.tree.intersection(tuple(box), objects=False))
            )
        return voxel_intersections

    def assert_voxelization(self, morphology, compartment_types):
        if len(morphology.cloud.get_voxels()) == 0:
            raise IncompleteMorphologyError(
                "Can't intersect without any {} in the {} morphology".format(
                    ", ".join(compartment_types), morphology.morphology_name
                )
            )


class FiberTransform(ConfigurableClass):
    @abc.abstractmethod
    def transform(self):
        pass


class QuiverTransform(FiberTransform):
    """
        QuiverTransform applies transformation to a Morphology based on an orientation field in a voxelized volume.
        Used for parallel fibers
    """

    def validate(self):
        if self.shared is True:
            raise ConfigurationError(
                "Attribute 'shared' can't be True for {} transformation".format(self.name)
            )

    def transform(self, point_cloud, orientation_data, volume_res):
        trans_vector_dx = [0, 0, 1]
        trans_vector_sx = [0, 0, -1]
        # Loop over all cells
        for cell in range(len(point_cloud)):
            # First 4 elements are the first compartment points (start and end) of each initial compartment of the 2 (parallel fiber) branches
            for comp in range(0, len(point_cloud[cell], 4)):
                # Right branch
                voxel_ind = point_cloud[cell][comp] / volume_res
                voxel_ind = voxel_ind.astype(int)
                orientation_vector = orientation_data[
                    :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                ]
                point_cloud[cell][comp + 1] = point_cloud[cell][comp] + np.cross(
                    orientation_vector, trans_vector_dx
                )
                # The new end is the nex start of the adjacent compartment
                point_cloud[cell][comp + 4] = point_cloud[cell][comp + 1]
                # Left branch
                voxel_ind = point_cloud[cell][comp + 2] / volume_res
                voxel_ind = voxel_ind.astype(int)
                orientation_vector = orientation_data[
                    :, voxel_ind[0], voxel_ind[1], voxel_ind[2]
                ]
                point_cloud[cell][comp + 3] = point_cloud[cell][comp + 2] + np.cross(
                    orientation_vector, trans_vector_sx
                )
                # The new end is the nex start of the adjacent compartment
                point_cloud[cell][comp + 6] = point_cloud[cell][comp + 3]

        return point_cloud

    # def plane_intersect(a, b):
    #     """
    #     a, b   4-tuples/lists
    #            Ax + By +Cz + D = 0
    #            A,B,C,D in order
    #
    #     output: 2 points on line of intersection, np.arrays, shape (3,)
    #     """
    #     a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
    #
    #     aXb_vec = np.cross(a_vec, b_vec)
    #
    #     A = np.array([a_vec, b_vec, aXb_vec])
    #     d = np.array([-a[3], -b[3], 0.]).reshape(3,1)
    #
    # # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
    #
    #     p_inter = np.linalg.solve(A, d).T
    #
    #     return p_inter[0], (p_inter + aXb_vec)[0]
    #
