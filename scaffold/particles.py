import numpy as np
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
        self.dimensions = np.array(dimensions)

class ParticleSystem:
    def fill(self, volume, voxels, particles):
        self.dimensions = len(volume)
        # Create a list of voxels where the particles can be placed.
        self.voxels = [ParticleVoxel(v[0], v[1]) for v in voxels]

        self.particles = []
        for particle_type in particles:
            radius = particle_type["radius"]
            placement_voxels = particle_type["voxels"]
            particle_count = particle_type["count"]
            # Generate a matrix with random positions for the particles
            # Add an extra dimension to determine in which voxels to place the particles
            placement_matrix = np.random.rand((particle_count, self.dimensions + 1))
            # Generate each particle
            for positions in placement_matrix:
                # Determine the voxel to be placed in.
                particle_voxel_id = int(positions[0] * (len(placement_voxels) - 1))
                particle_voxel = self.voxels[particle_voxel_id]
                # Translate the particle into the voxel based on the remaining dimensions
                particle_position = particle_voxel.origin + positions[1:] * particle_voxel.dimensions
                # Store the particle object
                self.add_particle(radius, particle_position)

    def add_particle(radius, position):
        self.particles.append(Particle(radius, position))

    def add_particles(radius, positions):
        for position in positions:
            self.add_particle(radius, position))

def plot_particle_system(system):
    trace = get_particles_trace(system.particles)
    fig = go.Figure(data=trace)
    fig.show()


def get_particles_trace(particles, dimensions=3, axes={'x': 0, 'y': 1, 'z': 2}, trace_kwargs=None):
    if trace_kwargs is None:
        trace_kwargs = {}
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
