from .strategy import Layered, PlacementStrategy
from ..particles import ParticleSystem
from ..exceptions import *
from ..reporting import report, warn


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
        system.fill(voxels, particles)
        # Raise a warning if no cells could be placed in the volume
        if len(system.particles) == 0:
            warn(
                "Did not place any {} cell in the {}!".format(cell_type.name, layer.name),
                PlacementWarning,
            )
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
        self.scaffold.place_cells(cell_type, layer, particle_positions)
