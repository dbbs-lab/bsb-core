from .strategy import Layered, PlacementStrategy
from ..helpers import assert_attr_array
import math, numpy as np
from ..exceptions import *
from ..reporting import report, warn


class Satellite(PlacementStrategy):
    """
        Implementation of the placement of cells in layers as satellites of existing cells

        Places cells as a satellite cell to each associated cell at a random distance
        depending on the radius of both cells.
    """

    defaults = {"per_planet": 1.0}

    def initialise(self, scaffold):
        super().initialise(scaffold)

    def validate(self):
        self.after = assert_attr_array(self, "planet_types", self.name)
        self.planet_cell_types = [self.scaffold.get_cell_type(p) for p in self.after]
        if self.layer is not None:
            warn(
                "Satellite cell '{}' specifies '{}' as its layer, but it will be ignored and the planet type's layer will be used instead.".format(
                    self.name, self.layer
                ),
                ConfigurationWarning,
            )
            self.layer = None

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
        # Initialise empty sattelite positions array
        satellites_pos = np.empty([0, 3])
        # Assemble the parallel arrays from all the planet cell types.
        for after_cell_type in after_cell_types:
            layer = after_cell_type.placement.layer_instance
            layer_min = layer.origin
            layer_max = layer.origin + layer.dimensions
            planet_cell_radius = after_cell_type.placement.radius
            planet_cells = self.scaffold.get_cells_by_type(after_cell_type.name)
            # Exit the placement of satellites if no corresponding planet after cells were created before
            if len(planet_cells) == 0:
                warn(
                    "Could not place any satellites for '{}' because no planet cells were created".format(
                        after_cell_type.name
                    ),
                    PlacementWarning,
                )
                continue
            planet_ids = planet_cells[:, 0]
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
            report(
                "Checking overlap and bounds of satellite {} cells...".format(
                    cell_type.name,
                ),
                level=3,
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
                warn(
                    "'{}' satellite cells out of '{}' have not been placed, due to overlapping or out of volume issues".format(
                        not_placed_num, len(planets_pos)
                    ),
                    PlacementWarning,
                )
            # Store the planet for each sattelite
            # NOTE: If we ever implement multiple sattelites per planet this implementation
            # will break.
            if not hasattr(scaffold, "_planets"):
                scaffold._planets = {}
            if cell_type.name not in scaffold._planets:
                scaffold._planets[cell_type.name] = []
            scaffold._planets[cell_type.name].extend(planet_ids)

        scaffold.place_cells(cell_type, layer, satellites_pos)
