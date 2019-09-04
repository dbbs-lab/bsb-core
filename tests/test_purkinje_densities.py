import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import ScaffoldIniConfig
from scaffold.scaffold import Scaffold
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scaffold.plotting import plotNetwork
import imageio


def get_placement_frame(cell_type, pos, angle, layer, squishy = 1.):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(title='Purkinje placement', xlabel='Angle = {}'.format(angle))
    color = cell_type.color
    ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],c=color)
    ax.set_xlim(layer.origin[0], layer.origin[0] + layer.dimensions[0])
    ax.set_ylim(layer.origin[1] - (squishy - 1.) * layer.dimensions[1], layer.origin[1] + layer.dimensions[1] * squishy)
    ax.set_zlim(layer.origin[2], layer.origin[2] + layer.dimensions[2])

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image

scaffoldConfig = ScaffoldIniConfig('test.ini')
scaffoldInstance = Scaffold(scaffoldConfig)
config = scaffoldInstance.configuration
layer = config.layers['Purkinje Layer']
pc = config.cell_types['Purkinje Cell']
steps = 400
angle_range = np.linspace(start=0.,stop=0.9,num=steps)
densities = np.empty((steps,2))
index = 0
frames = []

for angle in angle_range:
    scaffoldInstance.resetNetworkCache()
    pc.placement.angle = angle
    pc.placement.place(pc)
    pcCount = scaffoldInstance.cells_by_type['Purkinje Cell'].shape[0]
    density = pcCount / layer.width / layer.depth
    if pc.planarDensity is None:
        density /= layer.height
        densities[index, :] = [pc.density, density]
    else:
        densities[index, :] = [pc.planarDensity, density]
    index += 1
    frames.append(get_placement_frame(pc, scaffoldInstance.cells_by_type['Purkinje Cell'][[2,3,4]], angle, layer, 10.))

imageio.mimsave('./purkinje_debug.gif', frames, fps=24.)

plt.plot(angle_range, densities[:, 0], label='Expected density')
plt.plot(angle_range, densities[:, 1], label='Actual density')
plt.legend()
plt.show(block=True)
