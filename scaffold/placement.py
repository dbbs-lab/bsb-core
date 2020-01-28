import abc, math, random, numpy as np
from .helpers import ConfigurableClass, assert_attr_array
from scipy.spatial import distance
from scaffold.particles import Particle, ParticleSystem
from .functions import compute_circle, get_candidate_points, add_y_axis, exclude_index
from .exceptions import PlacementWarning, ConfigurationException


class PlacementStrategy(ConfigurableClass):
    def __init__(self, cell_type):
        super().__init__()
        self.cell_type = cell_type
        self.layer = None
        self.radius = None
        self.density = None
        self.planar_density = None
        self.placement_count_ratio = None
        self.density_ratio = None
        self.placement_relative_to = None
        self.count = None

    @abc.abstractmethod
    def place(self):
        pass

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    @abc.abstractmethod
    def get_placement_count(self):
        pass


class MightBeRelative:
    """
        Validation class for PlacementStrategies that can be configured relative to other
        cell types.
    """

    def validate(self):
        if self.placement_relative_to is not None:
            # Store the relation.
            self.relation = self.scaffold.configuration.cell_types[
                self.placement_relative_to
            ]
            if self.density_ratio is not None and self.relation.placement.layer is None:
                # A layer volume is required for relative density calculations.
                raise ConfigurationException(
                    "Cannot place cells relative to the density of a placement strategy that isn't tied to a layer."
                )

    def get_relative_count(self):
        # Get the placement count of the ratio cell type and multiply their count by the ratio.
        return int(
            self.relation.placement.get_placement_count() * self.placement_count_ratio
        )

    def get_relative_density_count(self):
        # Get the density of the ratio cell type and multiply it by the ratio.
        ratio = placement.placement_count_ratio
        n1 = self.relation.placement.get_placement_count()
        V1 = self.relation.placement.layer_instance.volume
        V2 = layer.volume
        return int(n1 * ratio * V2 / V1)


class MustBeRelative(MightBeRelative):
    """
        Validation class for PlacementStrategies that must be configured relative to other
        cell types.
    """

    def validate(self):
        if (
            not hasattr(self, "placement_relative_to")
            or self.placement_relative_to is None
        ):
            raise ConfigurationException(
                "The {} requires you to configure another cell type under `placement_relative_to`."
            )
        super().validate()


class Layered(MightBeRelative):
    """
        Class for placement strategies that depend on Layer objects.
    """

    def validate(self):
        super().validate()
        # Check if the layer is given and exists.
        config = self.scaffold.configuration
        if not hasattr(self, "layer"):
            raise Exception(
                "Required attribute 'layer' missing from {}".format(self.name)
            )
        if self.layer not in config.layers:
            raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
        self.layer_instance = self.scaffold.configuration.layers[self.layer]
        if hasattr(self, "y_restriction"):
            self.restriction_minimum = float(self.y_restriction[0])
            self.restriction_maximum = float(self.y_restriction[1])
        else:
            self.restriction_minimum = 0.0
            self.restriction_maximum = 1.0
        self.restriction_factor = self.restriction_maximum - self.restriction_minimum

    def get_placement_count(self):
        """
            Get the placement count proportional to the available volume in the layer
            times the cell type density.
        """
        layer = self.layer_instance
        available_volume = layer.available_volume
        placement = self.cell_type.placement
        if placement.count is not None:
            return int(placement.count)
        if placement.placement_count_ratio is not None:
            return self.get_relative_count()
        if placement.density_ratio is not None:
            return self.get_relative_density_count()
        if placement.planar_density is not None:
            # Calculate the planar density
            return int(layer.width * layer.depth * placement.planar_density)
        if hasattr(self, "restriction_factor"):
            # Add a restriction factor to the available volume
            return int(available_volume * self.restriction_factor * placement.density)
        # Default: calculate N = V * C
        return int(available_volume * placement.density)


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
            self.scaffold.warn(
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
                    if scaffold.configuration.verbosity > 0:
                        self.scaffold.warn(
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
                            scaffold.report(
                                "Only placed {} out of {} cells in sublayer {}".format(
                                    current_cell_count,
                                    cells_per_sublayer,
                                    sublayer_id + 1,
                                ),
                                3,
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
            scaffold.report(
                "Filling {} sublayer {}/{}...".format(
                    cell_type.name, sublayer_id + 1, n_sublayers
                ),
                3,
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


class Entities(Layered, PlacementStrategy):
    """
        Implementation of the placement of entities (e.g., mossy fibers) that do not have a
        a 3D position, but that need to be connected with other cells of the scaffold.
        MFs are 1/20 of the Glomeruli
    """

    entities = True

    def place(self):
        # Variables
        cell_type = self.cell_type
        scaffold = self.scaffold

        # Get the number of cells that belong in the available volume.
        n_cells_to_place = self.get_placement_count()
        if n_cells_to_place == 0:
            self.scaffold.warn(
                "Volume or density too low, no '{}' cells will be placed".format(
                    cell_type.name
                ),
                PlacementWarning,
            )

        scaffold.create_entities(cell_type, n_cells_to_place)


class ParallelArrayPlacement(Layered, PlacementStrategy):
    """
        Implementation of the placement of cells in parallel arrays.
    """

    casts = {
        "extension_x": float,
        "extension_z": float,
        "angle": lambda x: float(x) * 2 * math.pi / 360,
    }

    defaults = {"angle": 0.08726646259971647}  # 5 degrees

    required = ["extension_x", "extension_z", "angle"]

    def validate(self):
        # Check if the layer is given and exists.
        config = self.scaffold.configuration
        if not hasattr(self, "layer"):
            raise Exception("Required attribute Layer missing from {}".format(self.name))
        if self.layer not in config.layers:
            raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
        self.layer_instance = self.scaffold.configuration.layers[self.layer]

    def place(self):
        """
            Cell placement: Create a lattice of parallel arrays/lines in the layer's surface.
        """
        cell_type = self.cell_type
        layer = self.layer_instance
        radius = cell_type.placement.radius
        # Extension of a single array in the X dimension
        extension_x = self.extension_x
        # Add a random shift to the starting points of the arrays for variation.
        start_offset = np.random.rand() * extension_x
        # Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
        # They are placed in straight lines, tilted by a certain angle by adding a shifting value.
        x_positions = (
            np.arange(start=0.0, stop=layer.width, step=extension_x) + start_offset
        )
        if (
            x_positions.shape[0] == 0
        ):  # This only happens if the extension_x of a purkinje cell is larger than the simulation volume
            # Place a single row on a random position along the x axis
            x_positions = np.array([start_offset])
        # Amount of parallel arrays of cells
        n_arrays = x_positions.shape[0]
        # Number of cells
        n = self.get_placement_count()
        # Add extra cells to fill the lattice error volume which will be pruned
        n += int((n_arrays * extension_x % layer.width) / layer.width * n)
        # cells to distribute along the rows
        cells_per_row = round(n / n_arrays)
        # The rounded amount of cells that will be placed
        cells_placed = cells_per_row * n_arrays
        # Calculate the position of the cells along the z-axis.
        z_positions, z_axis_distance = np.linspace(
            start=0.0,
            stop=layer.depth - radius,
            num=cells_per_row,
            retstep=True,
            endpoint=False,
        )
        # Center the cell soma center to the middle of the unit cell
        z_positions += radius + z_axis_distance / 2
        # The length of the X axis rounded up to a multiple of the unit cell size.
        lattice_x = n_arrays * extension_x
        # The length of the X axis where cells can be placed in.
        bounded_x = lattice_x - radius * 2
        # Epsilon: jitter along the z-axis
        ϵ = self.extension_z / 2
        # Storage array for the cells
        cells = np.empty((cells_placed, 3))
        for i in np.arange(z_positions.shape[0]):
            # Shift the arrays at an angle
            angleShift = z_positions[i] * math.tan(self.angle)
            # Apply shift and offset
            x = x_positions + angleShift
            # Place the cells in a bounded lattice with a little modulus magic
            x = layer.origin[0] + x % bounded_x + radius
            # Place them at a uniformly random height throughout the layer.
            y = layer.origin[1] + np.random.uniform(
                radius, layer.height - radius, x.shape[0]
            )
            # Place the cells in their z-position with slight jitter
            z = layer.origin[2] + np.array(
                [
                    z_positions[i] + ϵ * (np.random.rand() - 0.5)
                    for _ in np.arange(x.shape[0])
                ]
            )
            # Store this stack's cells
            cells[(i * len(x)) : ((i + 1) * len(x)), 0] = x
            cells[(i * len(x)) : ((i + 1) * len(x)), 1] = y
            cells[(i * len(x)) : ((i + 1) * len(x)), 2] = z
        # Place all the cells in 1 stitch
        self.scaffold.place_cells(
            cell_type, layer, cells[cells[:, 0] < layer.width - radius]
        )


class Satellite(PlacementStrategy):
    """
        Implementation of the placement of cells in layers as satellites of existing cells

        Places cells as a satellite cell to each associated cell at a random distance
        depending on the radius of both cells.
    """

    defaults = {"per_planet": 1.0}

    def validate(self):
        self.after = assert_attr_array(self, "planet_types", self.name)
        self.planet_cell_types = [self.scaffold.get_cell_type(p) for p in self.after]

    def get_placement_count(self):
        """
            Takes the sum of the planets and multiplies it with the `per_planet` factor.
        """
        return (
            sum(
                [
                    planet.placement.get_placement_count()
                    for planet in self.planet_cell_types
                ]
            )
            * self.per_planet
        )

    def place(self):
        # Initialize
        cell_type = self.cell_type
        scaffold = self.scaffold
        config = scaffold.configuration
        radius_satellite = cell_type.placement.radius
        # Collect all planet cell types.
        after_cell_types = [
            self.scaffold.configuration.cell_types[type_after]
            for type_after in self.after
        ]
        all_satellites = np.empty((0))
        # Assemble the parallel arrays from all the planet cell types.
        for after_cell_type in after_cell_types:
            layer = after_cell_type.placement.layer_instance
            layer_min = layer.origin
            layer_max = layer.origin + layer.dimensions
            planet_cell_radius = after_cell_type.placement.radius
            planet_cells = self.scaffold.get_cells_by_type(after_cell_type.name)
            # Exit the placement of satellites if no corresponding planet after cells were created before
            if len(planet_cells) == 0:
                self.scaffold.warn(
                    "Could not place any satellites for '{}' because no planet cells were created".format(
                        after_cell_type.name
                    ),
                    PlacementWarning,
                )
                continue
            planet_ids =  planet_cells[:, 0]
            planets_pos = planet_cells[:, 2:5]
            planet_count = len(planets_pos)
            dist = np.empty((planet_count ** 2))
            for I in range(planet_count):
                for J in range(planet_count):
                    dist[I * planet_count + J] = np.linalg.norm(
                        planets_pos[I] - planets_pos[J]
                    )

            mean_dist_after_cells = np.mean(dist[np.nonzero(dist)])

            # Initialise satellite position array
            satellites_pos = np.empty([len(planet_cells), 3])
            scaffold.report(
                "Checking overlap and bounds of satellite {} cells...".format(
                    cell_type.name,
                ),
                3,
            )
            # To keep track of not placed particles that are not respecting the bounds or distances.
            not_placed_num = 0

            for i in reversed(range(len(planets_pos))):
                overlapping = True
                out = True
                attempts = 0
                # Place satellite and replace if it is overlapping or going out of the layer bounds
                while (overlapping or out_of_bounds) and attempts < 1000:
                    attempts += 1
                    alfa = np.random.uniform(0, 2 * math.pi)
                    beta = np.random.uniform(0, 2 * math.pi)
                    angles = np.array([np.cos(alfa), np.sin(alfa), np.sin(beta)])

                    # If we have only one planet and one satellite cell, we should place
                    # it near the planet without considering the mean distance of planets
                    if planet_count == 1:
                        distance = np.random.uniform(
                            (planet_cell_radius + radius_satellite),
                            (planet_cell_radius + radius_satellite) * 3,
                        )
                    else:
                        distance = np.random.uniform(
                            (planet_cell_radius + radius_satellite),
                            (
                                mean_dist_after_cells / 4
                                - planet_cell_radius
                                - radius_satellite
                            ),
                        )
                    # Calculate the satellite's position
                    satellites_pos[i] = distance * angles + planets_pos[i]

                    # Check overlapping: the distance of all planets to this satellite
                    # should be greater than the sum of their radii
                    distances_to_satellite = np.linalg.norm(
                        planets_pos - satellites_pos[i], axis=1
                    )
                    overlapping = not np.all(
                        distances_to_satellite > (planet_cell_radius + radius_satellite)
                    )

                    # Check out of bounds of layer: if any element of the satellite
                    # position is larger than the layer max or smaller than the layer min
                    # it is out of bounds.
                    out_of_bounds = np.any(
                        (satellites_pos[i] < layer_min) | (satellites_pos[i] > layer_max)
                    )

                if attempts >= 1000:
                    # The satellite cell cannot be placed. Remove it from the positions.
                    satellites_pos = np.delete(satellites_pos, (i), axis=0)
                    planet_ids = np.delete(planet_ids, (i), axis=0)
                    not_placed_num += 1

            if not_placed_num > 0:
                # Print warning that some satellite cells have not been placed
                self.scaffold.warn(
                    "'{}' satellite cells out of '{}' have not been placed, due to overlapping or out of volume issues".format(
                        not_placed_num, len(planets_pos)
                    ),
                    PlacementWarning,
                )

        scaffold.place_cells(cell_type, layer, satellites_pos)
        if not hasattr(scaffold, "_planets"):
            scaffold._planets = {}
        if cell_type.name not in scaffold._planets:
            scaffold._planets[cell_type.name] = []
        scaffold._planets[cell_type.name].extend(planet_ids)


class ParticlePlacement(Layered, PlacementStrategy):

    casts = {
        "prune": bool,
        "bounded": bool,
    }

    defaults = {
        "prune": True,
        "bounded": False,
    }

    def place(self):
        cell_type = self.cell_type
        layer = self.layer_instance
        origin = layer.origin.copy()
        # Shift voxel origin up based on y_restriction.
        origin[1] = layer.origin[1] + layer.thickness * self.restriction_minimum
        # Computing voxel thickness based on y_restriction
        volume = [layer.width, layer.thickness * self.restriction_factor, layer.depth]
        # Create a list of voxels with the current restricted layer as only voxel.
        voxels = [[origin, volume]]
        # Define the particles for the particle system.
        particles = [
            {
                "name": cell_type.name,
                "voxels": [0],
                "radius": cell_type.placement.radius,
                "count": self.get_placement_count(),
            }
        ]
        # Create and fill the particle system.
        system = ParticleSystem(track_displaced=True, scaffold=self.scaffold)
        system.fill(volume, voxels, particles)
        # Raise a warning if no cells could be placed in the volume
        if len(system.particles) == 0:
            self.scaffold.warn(
                "Did not place any {} cell in the {}!".format(cell_type.name, layer.name),
                PlacementWarning,
            )
            return

        # Find the set of colliding particles
        colliding = system.find_colliding_particles()
        if len(colliding) > 0:
            system.solve_collisions()
            if self.prune:
                number_pruned = system.prune(at_risk_particles=system.displaced_particles)
                self.scaffold.report(
                    "{} {} ({}%) cells pruned.".format(
                        number_pruned,
                        cell_type.name,
                        int((number_pruned / self.get_placement_count()) * 100),
                    )
                )
        particle_positions = system.positions
        self.scaffold.place_cells(cell_type, layer, particle_positions)
