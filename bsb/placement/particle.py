from .strategy import PlacementStrategy
from ..particles import ParticleSystem
from ..exceptions import *
from ..reporting import report, warn
from .. import config
import itertools, numpy as np


@config.node
class ParticlePlacement(PlacementStrategy):
    prune = config.attr(type=bool, default=True)
    bounded = config.attr(type=bool, default=False)
    restrict = config.attr(type=dict)

    def place(self, chunk, chunk_size, indicators):
        voxels = list(
            itertools.chain(
                *(p.chunk_to_voxels(chunk, chunk_size) for p in self.partitions)
            )
        )
        # Define the particles for the particle system.
        particles = [
            {
                "name": name,
                # Place particles in all voxels
                "voxels": list(range(len(voxels))),
                "radius": indicator.get_radius(),
                "count": int(indicator.guess(chunk, chunk_size)),
            }
            for name, indicator in indicators.items()
        ]
        # Create and fill the particle system.
        system = ParticleSystem(track_displaced=True, scaffold=self.scaffold)
        system.fill(voxels, particles)

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
                    total = indicator.guess(chunk, chunk_size)
                    pct = int((pruned / total) * 100)
                    report(f"{pruned} {name} ({pct}%) cells pruned.")

        for pt in system.particle_types:
            cell_type = self.scaffold.cell_types[pt["name"]]
            particle_positions = [p.position for p in system.particles if p.type is pt]
            positions = np.empty((len(particle_positions), 3))
            positions[:] = particle_positions
            self.scaffold.place_cells(cell_type, positions, chunk=chunk)
