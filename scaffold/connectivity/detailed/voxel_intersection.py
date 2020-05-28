import numpy as np
from ..strategy import ConnectionStrategy
from .shared import MorphologyStrategy
from ...helpers import DistributionConfiguration
from ...models import MorphologySet
from ...exceptions import *


class VoxelIntersection(ConnectionStrategy, MorphologyStrategy):
    """
        This strategy voxelizes morphologies into collections of cubes, thereby reducing
        the spatial specificity of the provided traced morphologies by grouping multiple
        compartments into larger cubic voxels. Intersections are found not between the
        seperate compartments but between the voxels and random compartments of matching
        voxels are connected to eachother. This means that the connections that are made
        are less specific to the exact morphology and can be very useful when only 1 or a
        few morphologies are available to represent each cell type.
    """

    casts = {
        "affinity": float,
        "contacts": DistributionConfiguration.cast,
        "voxels_pre": int,
        "voxels_post": int,
    }

    defaults = {
        "affinity": 1,
        "contacts": DistributionConfiguration.cast(1),
        "voxels_pre": 50,
        "voxels_post": 50,
    }

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
        to_compartments = self.to_cell_compartments[0]
        to_type = self.to_cell_types[0]
        from_placement_set = self.scaffold.get_placement_set(from_type.name)
        to_placement_set = self.scaffold.get_placement_set(to_type.name)
        from_cells = self.scaffold.get_cells_by_type(from_type.name)
        to_cells = self.scaffold.get_cells_by_type(to_type.name)

        # Load the morphology and voxelization data for the entrire morphology, for each cell type.
        from_morphology_set = MorphologySet(
            scaffold,
            from_type,
            from_placement_set,
            compartment_types=from_compartments,
            N=self.voxels_pre,
        )
        to_morphology_set = MorphologySet(
            scaffold,
            to_type,
            to_placement_set,
            compartment_types=to_compartments,
            N=self.voxels_post,
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
                # Only select a fraction of the total possible matches, based on how much
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
                contacts = round(self.contacts.sample())
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
