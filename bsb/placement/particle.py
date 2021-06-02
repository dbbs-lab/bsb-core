from .strategy import PlacementStrategy
from ..particles import ParticleSystem
from ..exceptions import *
from ..reporting import report, warn
from .. import config
import itertools


@config.node
class ParticlePlacement(PlacementStrategy):
    prune = config.attr(type=bool, default=True)
    bounded = config.attr(type=bool, default=False)

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
                report(
                    "{} {} ({}%) cells pruned.".format(
                        number_pruned,
                        cell_type.name,
                        int((number_pruned / self.get_placement_count()) * 100),
                    )
                )
        particle_positions = system.positions
        self.scaffold.place_cells(cell_type, particle_positions, chunk=chunk)
