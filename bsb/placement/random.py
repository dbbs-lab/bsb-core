import numpy as np

from .. import config
from ..exceptions import PackingError, PackingWarning
from ..reporting import report, warn
from ..voxels import VoxelSet
from .strategy import PlacementStrategy


class _VoxelBasedFiller:
    """
    Internal mixin for filler placement strategies
    """

    def _fill_system(self, chunk, indicators, check_pack=True):
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
        system = VolumeFiller(track_displaced=False, scaffold=self.scaffold, strat=self)
        system.fill(voxels, particles, check_pack=check_pack)
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
class RandomPlacement(PlacementStrategy, _VoxelBasedFiller):
    """
    Place cells in random positions.
    """

    def place(self, chunk, indicators):
        system = self._fill_system(chunk, indicators, check_pack=True)
        self._extract_system(system, chunk, indicators)


class Particle:
    def __init__(self, radius, position):
        self.radius = radius
        self.position = position


class Neighbourhood:
    def __init__(self, epicenter, neighbours, neighbour_radius, partners, partner_radius):
        self.epicenter = epicenter
        self.neighbours = neighbours
        self.neighbour_radius = neighbour_radius
        self.partners = partners
        self.partner_radius = partner_radius

    def get_overlap(self):
        overlap = 0
        neighbours = self.neighbours
        n_neighbours = len(neighbours)
        for partner in self.partners:
            for neighbour in self.neighbours:
                if partner.id == neighbour.id:
                    continue
                overlap -= min(
                    0,
                    distance(partner.position, neighbour.position)
                    - partner.radius
                    - neighbour.radius,
                )
        return overlap

    def colliding(self):
        neighbours = self.neighbours
        for partner in self.partners:
            for neighbour in self.neighbours:
                if partner.id == neighbour.id:
                    continue
                if (
                    distance(partner.position, neighbour.position)
                    - partner.radius
                    - neighbour.radius
                    < 0
                ):
                    return True
        return False


class ParticleVoxel:
    def __init__(self, origin, dimensions):
        self.origin = np.array(origin)
        self.size = np.array(dimensions)


class VolumeFiller:
    def __init__(self, track_displaced=False, scaffold=None, strat=None):
        self.particle_types = []
        self.voxels = []
        self.track_displaced = track_displaced
        self.scaffold = scaffold
        self.strat = strat

    def fill(self, voxels, particles, check_pack=True):
        """
        Fill a list of voxels with Particles.

        :param bsb.voxels.VoxelSet voxels: List of voxels in which to place the particles.
        :param List[dict] particles: List of dictionary for each particle to place.
            Each dictionary needs to contain the "name" (str), "radius" (float) of particle.
            It should also store the "count" of particle (int | List[int]) either in total or
            for each voxel and "voxels" (List[int]) which gives in which voxel the placement will
            append.
        :param bool check_pack: If True, will check the packing factor before placing particles in
            the voxels.
        :raise PackingError: If check_pack is True and the resulting packing factor is greater than
            0.4.
        """
        # Amount of spatial dimensions
        self.dimensions = voxels.get_raw(copy=False).shape[1]
        # Extend list of particle types in the system
        self.particle_types.extend(particles)
        # Max particle type radius
        self.max_radius = max([pt["radius"] for pt in self.particle_types])
        self.min_radius = min([pt["radius"] for pt in self.particle_types])
        # Set initial radius for collision/rearrangement to 2 times the largest particle type radius
        self.search_radius = self.max_radius * 2
        # Create a list of voxels where the particles can be placed.
        self.voxels.extend(
            ParticleVoxel(ldc, size)
            for ldc, size in zip(
                voxels.as_spatial_coords(copy=False), voxels.get_size_matrix(copy=False)
            )
        )
        if check_pack:
            pf = self.get_packing_factor()
            if self.strat is not None:
                strat_name = type(self.strat).__name__
            else:
                strat_name = "particle system"
            msg = f"Packing factor {round(pf, 2)}"
            if pf > 0.4:
                if pf > 0.64:
                    msg += " exceeds geometrical maximum packing for spheres (0.64)"
                elif pf > 0.4:
                    msg += f" too high to resolve with {strat_name}"

                count, pvol, vol = self._get_packing_factors()
                raise PackingError(
                    f"{msg}. Can not fit {round(count)} particles for a total of "
                    f"{round(pvol, 3)}μm³ micrometers into {round(vol, 3)}μm³."
                )
            elif pf > 0.2:
                warn(
                    f"{msg} is too high for good performance.",
                    PackingWarning,
                )
        # Reset particles
        self.particles = []
        for particle_type in self.particle_types:
            radius = particle_type["radius"]
            count = particle_type["count"]
            if count.size == 1:
                self._fill_global(particle_type)
            else:
                self._fill_per_voxel(particle_type)

    def _fill_per_voxel(self, particle_type):
        voxel_counts = particle_type["count"]
        radius = particle_type["radius"]
        if len(voxel_counts) != len(self.voxels):
            raise Exception(
                f"Particle system voxel mismatch. Given {len(voxel_counts)} expected {len(self.voxels)}"
            )
        for voxel, count in zip(self.voxels, voxel_counts):
            particle_type["placed"] = particle_type.get("placed", 0) + count
            placement_matrix = np.random.rand(count, self.dimensions)
            for in_voxel_pos in placement_matrix:
                particle_position = voxel.origin + in_voxel_pos * voxel.size
                self.add_particle(radius, particle_position, type=particle_type)

    def _fill_global(self, particle_type):
        particle_count = int(particle_type["count"])
        particle_type["placed"] = particle_type.get("placed", 0) + particle_count
        radius = particle_type["radius"]
        # Generate a matrix with random positions for the particles
        # Add an extra dimension to determine in which voxels to place the particles
        placement_matrix = np.random.rand(particle_count, self.dimensions + 1)
        # Generate each particle
        for row in placement_matrix:
            # Determine the voxel to be placed in.
            voxel_id = int(row[0] * len(self.voxels))
            voxel = self.voxels[voxel_id]
            # Translate the particle into the voxel based on the remaining dimensions
            particle_position = voxel.origin + row[1:] * voxel.size
            # Store the particle object
            self.add_particle(radius, particle_position, type=particle_type)

    @property
    def positions(self):
        x = np.array([p.position for p in self.particles])
        return x

    def add_particle(self, radius, position, type=None):
        particle = Particle(radius, position)
        particle.id = len(self.particles)
        particle.type = type
        self.particles.append(particle)

    def add_particles(self, radius, positions, type=None):
        for position in positions:
            self.add_particle(radius, position, type=type)

    def remove_particles(self, particles_id):
        # Remove particles with a certain id
        for index in sorted(particles_id, reverse=True):
            del self.particles[index]

    def get_packing_factor(self, particles=None, volume=None):
        """
        Calculate the packing factor of the volume where particles will be placed.
        It corresponds to the ratio of the sum of the particles' volume over the volume itself.

        :param List[bsb.placement.particle.Particle] | None particles: List of Particle to place.
            If None, it will use the ParticleSystem particle_types list.
        :param float | None volume: Size of the volume in which the particles will be placed.
            If None, it will use the total volume of the voxels of the ParticleSystem.
        :return: Packing factor
        :rtype: float
        """
        if particles is None:
            particles_volume = sum(
                np.sum(p["count"]) * sphere_volume(p["radius"])
                for p in self.particle_types
            )
        else:
            particles_volume = sum(sphere_volume(p.radius) for p in particles)
        if volume is None:
            volume = sum(
                sum(np.prod(v.size) for v in np.array(self.voxels)[np.array(p["voxels"])])
                for p in self.particle_types
            )
        return particles_volume / volume

    def _get_packing_factors(self, particles=None, volume=None):
        if particles is None:
            particles_volume = np.sum(
                [p["count"] * sphere_volume(p["radius"]) for p in self.particle_types]
            )
            particles_count = np.sum([p["count"] for p in self.particle_types])
        else:
            particles_volume = np.sum([sphere_volume(p.radius) for p in particles])
            particles_count = len(particles)
        if volume is None:
            volume = np.sum([np.prod(v.size) for v in self.voxels])
        return particles_count, particles_volume, volume


def sphere_volume(radius):
    return 4 / 3 * np.pi * radius**3


def distance(a, b):
    return np.sqrt(np.sum((b - a) ** 2))


__all__ = ["RandomPlacement"]
