import numpy as np
from ..strategy import ConnectionStrategy
from .shared import MorphologyStrategy
from ...models import MorphologySet
from ...exceptions import *


class VoxelIntersection(ConnectionStrategy, MorphologyStrategy):
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
        to_compartments = self.to_cell_compartments[0]
        to_type = self.to_cell_types[0]
        from_cells = self.scaffold.get_cells_by_type(from_type.name)
        to_cells = self.scaffold.get_cells_by_type(to_type.name)

        # Load the morphology and voxelization data for the entrire morphology, for each cell type.
        from_morphology_set = MorphologySet(
            scaffold, from_type, from_cells, compartment_types=from_compartments
        )
        to_morphology_set = MorphologySet(
            scaffold, to_type, to_cells, compartment_types=to_compartments
        )
        joined_map = (
            from_morphology_set._morphology_map + to_morphology_set._morphology_map
        )
        joined_map_offset = len(from_morphology_set._morphology_map)

        # For every postsynaptic cell, derive the box incorporating all voxels,
        # and store that box in the tree, to later find intersections with that cell.
        for i, (to_cell, morphology) in enumerate(to_morphology_set):
            to_offset = np.concatenate((to_cell.position, to_cell.position))
            if len(morphology.cloud.get_voxels()) == 0:
                raise IncompleteMorphologyError(
                    "Can't intersect without any {} in the {} morphology".format(
                        ", ".join(to_compartments), morphology.morphology_name
                    )
                )
            to_box = morphology.cloud.get_voxel_box()
            to_cell_tree.insert(i, tuple(to_box + to_offset))

        # For each presynaptic cell, find all postsynaptic cells that its outer
        # box intersects with.
        connections_out = []
        compartments_out = []
        morphologies_out = []
        for from_cell, from_morpho in from_morphology_set:
            from_box = from_morpho.cloud.get_voxel_box()
            from_map = from_morpho.cloud.map
            this_box = tuple(
                from_box + np.concatenate((from_cell.position, from_cell.position))
            )
            cell_intersections = list(to_cell_tree.intersection(this_box, objects=False))
            for partner in cell_intersections:
                to_cell, to_morpho = to_morphology_set[partner]
                to_map = to_morpho.cloud.map
                voxel_intersections = self.intersect_clouds(
                    from_morpho.cloud,
                    to_morpho.cloud,
                    from_cell.position,
                    to_cell.position,
                )
                # Find non-empty lists: these voxels have intersections
                intersecting_from_voxels = np.nonzero(voxel_intersections)[0]
                if not len(intersecting_from_voxels):
                    continue
                # Data structure to contain the compartment pairs of this cell pair.
                cell_pair_compartment_pairs = {}
                for from_voxel_id in intersecting_from_voxels:
                    # Get the list of voxels that the from_voxel intersects with.
                    intersecting_voxels = voxel_intersections[from_voxel_id]
                    to_voxel_candidates = []
                    for to_voxel_id in intersecting_voxels:
                        # Store all of the compartments in the to_voxel as
                        # possible candidates for this cell pair's connection
                        to_voxel_candidates.extend(to_map[to_voxel_id])
                    cell_pair_compartment_pairs[from_voxel_id] = to_voxel_candidates
                # Weigh the random sampling by the amount of compartment pairs
                voxel_weights = list(
                    map(
                        lambda item: len(from_map[item[0]]) * len(item[1]),
                        cell_pair_compartment_pairs.items(),
                    )
                )
                weight_sum = sum(voxel_weights)
                voxel_weights = [w / weight_sum for w in voxel_weights]
                pair_items = list(cell_pair_compartment_pairs.items())
                random_pair_id = np.random.choice(
                    range(len(pair_items)), 1, p=voxel_weights
                )[0]
                random_voxel_id, to_compartments = pair_items[random_pair_id]
                # Pick a random from and to compartment of the chosen voxel pair
                from_compartment = np.random.choice(from_map[random_voxel_id], 1)[0]
                to_compartment = np.random.choice(to_compartments, 1)[0]
                compartments_out.append([from_compartment, to_compartment])
                morphologies_out.append(
                    [from_morpho._set_index, joined_map_offset + to_morpho._set_index]
                )
                connections_out.append([from_cell.id, to_cell.id])

        if len(connections_out) > 0:
            self.scaffold.connect_cells(
                self,
                np.array(connections_out),
                morphologies=np.array(morphologies_out),
                compartments=np.array(compartments_out),
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
