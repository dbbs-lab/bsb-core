from .strategy import PlacementStrategy
import math, numpy as np
from ..exceptions import *
from ..reporting import report, warn
from .. import config
from ..config import types, refs
from itertools import chain
from .indicator import PlacementIndicator


class SatelliteIndicator(PlacementIndicator):
    def guess(self, chunk=None):
        planet_types = self._strat.planet_types

        def strats_of(pt):
            return self._strat.scaffold.get_placement_of(pt)

        def count_of(strat, pt):
            return strat.get_indicators()[pt.name].guess(chunk)

        return (
            sum(
                sum(count_of(strat, pt) for strat in strats_of(pt)) for pt in planet_types
            )
            * self._strat.per_planet
        )


@config.node
class Satellite(PlacementStrategy):
    """
    Implementation of the placement of cells in layers as satellites of existing cells

    Places cells as a satellite cell to each associated cell at a random distance
    depending on the radius of both cells.
    """

    per_planet = config.attr(type=float, default=1.0)
    planet_types = config.reflist(refs.cell_type_ref, required=True)
    partitions = config.reflist(refs.regional_ref)
    indicator_class = SatelliteIndicator

    def boot(self):
        # Use the planet type partitions as our partitions
        placements = self.scaffold.get_placement_of(*self.planet_types)
        self.partitions = list(chain(*(p.partitions for p in placements)))

    def get_after(self):
        return self.scaffold.get_placement_of(*self.planet_types)

    def place(self, chunk, indicators):
        for indicator in indicators.values():
            self.place_type(chunk, indicator)

    def place_type(self, chunk, indicator):
        # STRATEGY DISABLED DURING v4 REWORK
        return
        cell_type = indicator.cell_type
        scaffold = self.scaffold
        radius_satellite = indicator.get_radius()
        # Assemble the parallel arrays from all the planet cell types.
        for after_cell_type in self.planet_types:
            planets_pos = after_cell_type.get_placement_set().load_positions()

        for after_cell_type in self.planet_types:
            layer = after_cell_type.placement.partitions[0]
            layer_min = layer.boundaries.ldc
            layer_max = layer.boundaries.mdc
            planet_cell_radius = after_cell_type.spatial.radius
            planet_cells = self.scaffold.get_placement_set(after_cell_type.name)
            planet_cells.load_chunk(chunk)
            # Exit the placement of satellites if no corresponding planet after cells were created before
            if len(planet_cells) == 0:
                warn(
                    "Could not place any satellites for '{}' because no planet cells were created".format(
                        after_cell_type.name
                    ),
                    PlacementWarning,
                )
                continue
            planet_ids = planet_cells.load_identifiers()
            planets_pos = planet_cells.load_positions()
            planet_count = len(planets_pos)
            dist = np.empty((planet_count**2))
            for I in range(planet_count):
                for J in range(planet_count):
                    dist[I * planet_count + J] = np.linalg.norm(
                        planets_pos[I] - planets_pos[J]
                    )

            mean_dist_after_cells = np.mean(dist[np.nonzero(dist)])

            # Initialise satellite position array
            self.satellites_pos = np.empty([len(planet_cells), 3])
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
                    overlapping = (
                        np.any(
                            distances_to_satellite
                            < (planet_cell_radius + radius_satellite)
                        )
                        or np.any()
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

        self.place_cells(indicator, satellites_pos, chunk=chunk)
