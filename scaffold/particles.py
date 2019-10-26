import numpy as np
from sklearn.neighbors import KDTree
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    pass

class Particle:
    def __init__(self, radius, position):
        self.radius = radius
        self.position = position

class ParticleVoxel:
    def __init__(self, origin, dimensions):
        self.origin = np.array(origin)
        self.size = np.array(dimensions)

class ParticleSystem:
    def __init__(self):
        self.particle_types = []
        self.voxels = []

    def fill(self, volume, voxels, particles):
        # Amount of spatial dimensions
        self.dimensions = len(volume)
        #
        self.size = volume
        # Extend list of particle types in the system
        self.particle_types.extend(particles)
        # Max particle type radius
        self.max_radius = max([pt["radius"] for pt in self.particle_types])
        # Set initial radius for collision/rearrangement to 2 times the largest particle type radius
        self.search_radius = self.max_radius * 2
        # Create a list of voxels where the particles can be placed.
        self.voxels.extend([ParticleVoxel(v[0], v[1]) for v in voxels])
        # Reset particles
        self.particles = []
        for particle_type in self.particle_types:
            radius = particle_type["radius"]
            placement_voxels = particle_type["voxels"]
            particle_count = particle_type["count"]
            # Generate a matrix with random positions for the particles
            # Add an extra dimension to determine in which voxels to place the particles
            placement_matrix = np.random.rand(particle_count, self.dimensions + 1)
            # Generate each particle
            for positions in placement_matrix:
                # Determine the voxel to be placed in.
                particle_voxel_id = int(positions[0] * len(placement_voxels))
                particle_voxel = self.voxels[particle_voxel_id]
                # Translate the particle into the voxel based on the remaining dimensions
                particle_position = particle_voxel.origin + positions[1:] * particle_voxel.size
                # Store the particle object
                self.add_particle(radius, particle_position)

    def freeze(self):
        self.positions = [p.position for p in self.particles]
        self.radii = [p.radius for p in self.particles]
        self.tree = KDTree(self.positions)

    def find_collisions(self):
        collisions = []
        if not hasattr(self, "tree"):
            self.freeze()
        # Do an O(n * log(n)) search of all particles by the maximum radius
        neighbours = self.tree.query_radius(self.positions, r=self.search_radius, return_distance=True)
        neighbour_ids = neighbours[0]
        neighbour_distances = neighbours[1]
        for i in range(len(neighbour_ids)):
            distances = neighbour_distances[i]
            if len(distances) == 1: # No neighbours
                continue
            ids = neighbour_ids[i]
            particle = self.particles[i]
            for j in range(len(ids)):
                neighbour_id = ids[j]
                if neighbour_id == i:
                    continue
                neighbour = self.particles[neighbour_id]
                min_radius = particle.radius + neighbour.radius
                if distances[j] <= min_radius:
                    particle.colliding = True
                    neighbour.colliding = True
                    collisions.append([particle, neighbour])
        self.collisions = collisions
        return collisions

    def add_particle(self, radius, position):
        particle = Particle(radius, position)
        particle.id = len(self.particles)
        self.particles.append(particle)

    def add_particles(self, radius, positions):
        for position in positions:
            self.add_particle(radius, position)

    def deintersect(self, nearest_neighbours=None):
        if nearest_neighbours is None:
            nearest_neighbours = self.estimate_nearest_neighbours()

    def get_packing_factor(self):
        particles_volume = np.sum([p["count"] * (4 / 3 * np.pi * p["radius"] ** 3) for p in self.particle_types])
        total_volume = np.product(self.size)
        return particles_volume / total_volume

def plot_particle_system(system):
    trace = get_particles_trace(system.particles)
    fig = go.Figure(data=trace)
    if system.dimensions == 3:
        fig.update_layout(scene_aspectmode='cube')
        fig.layout.scene.xaxis.range = [0., system.size[0]]
        fig.layout.scene.yaxis.range = [0., system.size[1]]
        fig.layout.scene.zaxis.range = [0., system.size[2]]
    fig.show()


def get_particles_trace(particles, dimensions=3, axes={'x': 0, 'y': 1, 'z': 2}, **kwargs):
    trace_kwargs = {"mode": "markers", "marker": {"color": "rgba(100, 100, 100, 1)", "size": 1}}
    trace_kwargs.update(kwargs)
    if dimensions > 3:
        raise Exception("Maximum 3 dimensional plots. Unless you have mutant eyes.")
    elif dimensions == 3:
        return go.Scatter3d(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            y=list(map(lambda p: p.position[axes['y']], particles)),
            z=list(map(lambda p: p.position[axes['z']], particles)),
            **trace_kwargs
        )
    elif dimensions == 2:
        return go.Scatter(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            y=list(map(lambda p: p.position[axes['y']], particles)),
            **trace_kwargs
        )
    elif dimensions == 1:
        return go.Scatter(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            **trace_kwargs
        )
