from .strategy import Layered, PlacementStrategy
import random, numpy as np
from ..functions import get_candidate_points, add_y_axis, exclude_index
from ..reporting import report, warn
from scipy.spatial import distance
from ..exceptions import *


class LayeredRandomWalk(Layered, PlacementStrategy):
    """
        Implementation of the placement of cells in sublayers via a self avoiding random walk.
    """

    casts = {"distance_multiplier_min": float, "distance_multiplier_max": float}

    defaults = {"distance_multiplier_min": 0.75, "distance_multiplier_max": 1.25}

    def validate(self):
        super().validate()

    def place(self):
        """
            The LayeredRandomWalk subdivides the available volume into
            sublayers and distributes cells into each sublayer using a
            self-avoiding random walk.
        """
        # Variables
        cell_type = self.cell_type
        scaffold = self.scaffold
        config = scaffold.configuration
        layer = self.layer_instance
        layer_thickness = self.get_restricted_thickness()
        # Virtual layer origin point that applies the Y-Restriction used for example by basket and stellate cells.
        restricted_origin = np.array(
            [
                layer.origin[0],
                layer.origin[1] + layer.thickness * self.restriction_minimum,
                layer.origin[2],
            ]
        )
        # Virtual layer dimensions that apply the Y-Restriction used for example by basket and stellate cells.
        restricted_dimensions = np.array(
            [layer.dimensions[0], layer_thickness, layer.dimensions[2]]
        )
        cell_radius = cell_type.placement.radius
        cell_bounds = np.column_stack(
            (
                restricted_origin + cell_radius,
                restricted_origin + restricted_dimensions - cell_radius,
            )
        )
        # Get the number of cells that belong in the available volume.
        n_cells_to_place = self.get_placement_count()
        if n_cells_to_place == 0:
            warn(
                "Volume or density too low, no '{}' cells will be placed".format(
                    cell_type.name
                ),
                PlacementWarning,
            )
            n_sublayers = 1
            cell_type.ϵ = 0.0
        else:
            # Calculate the volume available per cell
            cell_type.placement_volume = (
                layer.available_volume * self.restriction_factor / n_cells_to_place
            )
            # Calculate the radius of that volume's sphere
            cell_type.placement_radius = (0.75 * cell_type.placement_volume / np.pi) ** (
                1.0 / 3.0
            )
            # Calculate the cell epsilon: This is the length of the 'spare space' a cell has inside of its volume
            cell_type.ϵ = cell_type.placement_radius - cell_radius
            # Calculate the amount of sublayers
            n_sublayers = int(
                np.round(layer_thickness / (1.5 * cell_type.placement_radius))
            )
        ## Sublayer partitioning
        partitions = self.partition_layer(n_sublayers)
        # Adjust partitions for cell radius.
        partitions = partitions + np.array([cell_radius, -cell_radius])

        ## Placement
        min_ϵ = self.distance_multiplier_min * cell_type.ϵ
        max_ϵ = self.distance_multiplier_max * cell_type.ϵ
        cells_per_sublayer = max(1, int(np.round(n_cells_to_place / n_sublayers)))

        layer_cell_positions = np.empty((0, 3))
        previously_placed_cells = scaffold.cells_by_layer[layer.name][:, [2, 3, 4]]
        previously_placed_types = np.array(
            scaffold.cells_by_layer[layer.name][:, 1], dtype=int
        )
        other_cell_type_radii = np.array(
            list(
                map(
                    lambda type: scaffold.configuration.cell_types[type].placement.radius,
                    scaffold.configuration.cell_type_map,
                )
            )
        )
        if len(previously_placed_cells) > 0:
            previously_placed_min_dist = (
                other_cell_type_radii[previously_placed_types] + cell_radius
            )
        else:
            previously_placed_min_dist = np.empty((0))
        other_cell_type_count = previously_placed_min_dist.shape[0]

        for sublayer_id in np.arange(n_sublayers):
            if cells_per_sublayer == 0:
                continue
            sublayer_id = int(sublayer_id)
            sublayer_floor = partitions[sublayer_id, 0]
            sublayer_roof = partitions[sublayer_id, 1]
            sublayer_attempts = 0

            # Generate the first cell's position.
            starting_position = np.array(
                (
                    np.random.uniform(cell_bounds[0, 0], cell_bounds[0, 1]),  # X
                    np.random.uniform(cell_bounds[1, 0], cell_bounds[1, 1]),  # Y
                    np.random.uniform(cell_bounds[2, 0], cell_bounds[2, 1]),  # Z
                )
            )
            planar_start = [starting_position[0], starting_position[2]]  # X & Z
            # Get all possible new cell positions
            planar_candidates = get_candidate_points(
                planar_start, cell_radius, cell_bounds, min_ϵ, max_ϵ
            )
            # If there are no possible points, force the cell position to be in the middle of surface
            if planar_candidates.shape[0] == 0:
                starting_position = np.array(
                    [
                        cell_bounds[0, 0]
                        + (cell_bounds[0, 1] - cell_bounds[0, 0]) / 2.0,  # X
                        np.random.uniform(sublayer_floor, sublayer_roof),  # Y
                        cell_bounds[2, 0]
                        + (cell_bounds[2, 1] - cell_bounds[2, 0]) / 2.0,  # Z
                    ]
                )
                planar_candidates = get_candidate_points(
                    planar_start, cell_radius, cell_bounds, min_ϵ, max_ϵ
                )
                if planar_candidates.shape[0] == 0:
                    warn(
                        "Could not place a single cell in {} {} starting from the middle of the simulation volume: Maybe the volume is too low or cell radius/epsilon too big. Sublayer skipped!".format(
                            layer.name, sublayer_id
                        ),
                        PlacementWarning,
                    )
                    continue
            placed_positions = np.array([starting_position])
            planar_placed_positions = np.array([starting_position[[0, 2]]])
            full_coords = add_y_axis(
                planar_placed_positions, sublayer_floor, sublayer_roof
            )
            good_points_store = [np.copy(full_coords)]
            last_position = starting_position
            for current_cell_count in np.arange(1, cells_per_sublayer, dtype=int):
                # Subtract failed placement attempts from loop counter
                current_cell_count = current_cell_count - sublayer_attempts
                planar_candidates, rnd_ϵ = get_candidate_points(
                    last_position[[0, 2]],
                    cell_radius,
                    cell_bounds,
                    min_ϵ,
                    max_ϵ,
                    return_ϵ=True,
                )
                full_coords = add_y_axis(planar_candidates, sublayer_floor, sublayer_roof)
                inter_cell_soma_dist = cell_radius * 2 + rnd_ϵ
                sublayer_distances = distance.cdist(
                    planar_candidates, planar_placed_positions
                )
                good_indices = list(
                    np.where(
                        np.sum(sublayer_distances > inter_cell_soma_dist, axis=1)
                        == current_cell_count
                    )[0]
                )
                planar_candidates = planar_candidates[good_indices]
                full_coords = full_coords[good_indices]
                layer_distances = distance.cdist(full_coords, previously_placed_cells)
                good_indices = list(
                    np.where(
                        np.sum(layer_distances > previously_placed_min_dist, axis=1)
                        == other_cell_type_count
                    )[0]
                )
                if len(good_indices) == 0:
                    max_attempts = len(good_points_store)
                    for attempt in range(max_attempts):
                        store_id = np.random.randint(max_attempts - attempt)
                        planar_candidates = good_points_store[store_id][:, [0, 2]]
                        full_coords = good_points_store[store_id]
                        rnd_ϵ = np.random.uniform(min_ϵ, max_ϵ)
                        inter_cell_soma_dist = cell_radius * 2 + rnd_ϵ
                        sublayer_distances = distance.cdist(
                            planar_candidates, planar_placed_positions
                        )
                        good_indices = list(
                            np.where(
                                np.sum(sublayer_distances > inter_cell_soma_dist, axis=1)
                                == current_cell_count
                            )[0]
                        )
                        planar_candidates = planar_candidates[good_indices]
                        full_coords = full_coords[good_indices]
                        layer_distances = distance.cdist(
                            full_coords, previously_placed_cells
                        )
                        good_indices = list(
                            np.where(
                                np.sum(
                                    layer_distances > previously_placed_min_dist, axis=1
                                )
                                == other_cell_type_count
                            )[0]
                        )
                        if len(good_indices) > 0:
                            random_index = random.sample(good_indices, 1)[0]
                            candidate = full_coords[random_index]
                            placed_positions = np.vstack([placed_positions, candidate])
                            planar_placed_positions = np.vstack(
                                [planar_placed_positions, candidate[[0, 2]]]
                            )
                            last_position = candidate
                            break
                        else:
                            good_points_store = exclude_index(good_points_store, store_id)
                    if len(good_indices) == 0:
                        if sublayer_attempts < 10:
                            sublayer_attempts += 1
                            # Try again from a random position
                            last_position = np.array(
                                [
                                    cell_bounds[0, 0]
                                    + (cell_bounds[0, 1] - cell_bounds[0, 0]) / 2.0,  # X
                                    np.random.uniform(sublayer_floor, sublayer_roof),  # Y
                                    cell_bounds[2, 0]
                                    + (cell_bounds[2, 1] - cell_bounds[2, 0]) / 2.0,  # Z
                                ]
                            )
                        else:
                            report(
                                "Only placed {} out of {} cells in sublayer {}".format(
                                    current_cell_count,
                                    cells_per_sublayer,
                                    sublayer_id + 1,
                                ),
                                level=3,
                            )
                            break
                else:
                    random_index = random.sample(good_indices, 1)[0]
                    new_position = full_coords[random_index]
                    placed_positions = np.vstack([placed_positions, new_position])
                    planar_placed_positions = np.vstack(
                        [planar_placed_positions, new_position[[0, 2]]]
                    )

                    good_points_store.append(full_coords[good_indices])
                    last_position = new_position

            layer_cell_positions = np.concatenate(
                (layer_cell_positions, placed_positions)
            )
            report(
                "Filling {} sublayer {}/{}...".format(
                    cell_type.name, sublayer_id + 1, n_sublayers
                ),
                level=3,
                ongoing=True,
            )

        scaffold.place_cells(cell_type, layer, layer_cell_positions)

    def partition_layer(self, n_sublayers):
        # Allow restricted placement along the Y-axis.
        yMin = self.restriction_minimum
        layer_thickness = self.get_restricted_thickness()
        sublayerHeight = layer_thickness / n_sublayers
        # Divide the Y axis into equal pieces
        sublayerYs = np.linspace(sublayerHeight, layer_thickness, n_sublayers)
        # Add the bottom of the lowest layer and translate all the points by the layer's Y position, keeping the Y restriction into account
        sublayerYs = (
            np.insert(sublayerYs, 0, 0)
            + self.layer_instance.origin[1]
            + yMin * self.layer_instance.thickness
        )
        # Create pairs of points on the Y axis corresponding to the bottom and ceiling of each sublayer partition
        sublayerPartitions = np.column_stack([sublayerYs, np.roll(sublayerYs, -1)])[:-1]
        return sublayerPartitions

    def get_restricted_thickness(self):
        return self.layer_instance.thickness * (
            self.restriction_maximum - self.restriction_minimum
        )
