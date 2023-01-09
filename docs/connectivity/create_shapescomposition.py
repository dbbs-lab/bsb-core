import numpy as np
from bsb.core import from_storage
import plotly.graph_objects as go
from bsb.connectivity.point_cloud.geometric_shapes import (
    ShapesComposition,
    Sphere,
    Cylinder,
    Cone,
)
from bsb.plotting import plot_shape_wireframe

# Read the network and the morphology
network = from_storage("HumanMorphologies.hdf5")
mr = network.morphologies

# We create
cone_base_center = np.array([0, 0, 40], dtype=np.float64)
cone_base_radius = 50.0
cylinder_radius = 25.0
soma_radius = 10.0
soma_position = np.array([0, 0, 0], dtype=np.float64)
axon_endpoint = np.array([0, 0, -40], dtype=np.float64)

# We create a ShapesComposition object
sc = ShapesComposition(20)

# Build the soma
sc.add_shape(Sphere(soma_position, soma_radius), ["soma"])

# Build the axon
sc.add_shape(Cylinder(axon_endpoint, cylinder_radius, soma_position), ["axon"])

# Build the dendritic tree
sc.add_shape(Cone(soma_position, cone_base_center, cone_base_radius), ["dendrites"])

# Save point cloud
sc.save_to_file("point_cloud.pck")

# Plot the result
fig = go.Figure()
x, y, z = sc.generate_wireframe()
plot_shape_wireframe(x, y, z, fig)
fig.show()
