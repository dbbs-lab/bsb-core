from .strategy import PlacementStrategy
from ..voxels import VoxelSet
from ..particles import ParticleSystem
from ..exceptions import *
from ..reporting import report, warn
from .. import config
import itertools, numpy as np


class _VoxelBasedParticleSystem:
    """
    Internal mixin for particle system placement strategies
    """

    def _fill_system(self, chunk, indicators):
        voxels = VoxelSet.concatenate(
            *(p.chunk_to_voxels(chunk) for p in self.partitions)
        )
        # Define the particles for the particle system.
        particles = [
            {
                "name": name,
                # Place particles in all voxels
                "voxels": list(range(len(voxels))),
                "radius": indicator.get_radius(),
                # Indicator guesses either a global value, or a per voxel value.
                "count": indicator.guess(chunk, voxels),
            }
            for name, indicator in indicators.items()
        ]
        # Create and fill the particle system.
        system = ParticleSystem(track_displaced=True, scaffold=self.scaffold)
        system.fill(voxels, particles)
        return system

    def _extract_system(self, system, chunk, indicators):
        if len(system.particles) == 0:
            return

        for pt in system.particle_types:
            cell_type = self.scaffold.cell_types[pt["name"]]
            indicator = indicators[pt["name"]]
            particle_positions = [p.position for p in system.particles if p.type is pt]
            if len(particle_positions) == 0:
                continue
            positions = np.empty((len(particle_positions), 3))
            positions[:] = particle_positions
            report(f"Placing {len(positions)} {cell_type.name} in {chunk}", level=3)
            self.place_cells(indicator, positions, chunk)


@config.node
class RandomPlacement(PlacementStrategy, _VoxelBasedParticleSystem):
    """
    Place cells in random positions.
    """

    def place(self, chunk, indicators):
        system = self._fill_system(chunk, indicators)
        self._extract_system(system, chunk, indicators)


@config.node
class ParticlePlacement(PlacementStrategy, _VoxelBasedParticleSystem):
    """
    Place cells in random positions, then have them repel each other until there is no
    overlap.
    """

    prune = config.attr(type=bool, default=True)
    bounded = config.attr(type=bool, default=False)
    restrict = config.attr(type=dict)

    def place(self, chunk, indicators):
        system = self._fill_system(chunk, indicators)
        if len(system.particles) == 0:
            return
        # Find the set of colliding particles
        colliding = system.find_colliding_particles()
        if len(colliding) > 0:
            system.solve_collisions()
            if self.prune:
                number_pruned, pruned_per_type = system.prune(
                    at_risk_particles=system.displaced_particles
                )
                for name, indicator in indicators.items():
                    pruned = pruned_per_type[name]
                    total = [
                        pt.get("placed", None)
                        for pt in system.particle_types
                        if pt["name"] == name
                    ][0]
                    if not total:
                        pct = 0
                    else:
                        pct = int((pruned / total) * 100)
                    report(f"{pruned} {name} ({pct}%) cells pruned.")
        self._extract_system(system, chunk, indicators)
