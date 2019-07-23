import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scaffold.config import ScaffoldIniConfig
from scaffold.scaffold import Scaffold
import matplotlib.pyplot as plt
import numpy as np
from scaffold.plotting import plotNetwork

scaffoldConfig = ScaffoldIniConfig('test.ini')
scaffoldInstance = Scaffold(scaffoldConfig)
config = scaffoldInstance.configuration
layer = config.Layers['Purkinje Layer']
pc = config.CellTypes['Purkinje Cell']
angle_range = np.linspace(start=0.,stop=0.9,num=400)
densities = np.empty((400,2))
index = 0
show_once = True
pc.placement.extension_x = 130
for angle in angle_range:
    scaffoldInstance.resetNetworkCache()
    pc.placement.angle = angle
    pc.placement.place(pc)
    pcCount = scaffoldInstance.CellsByType['Purkinje Cell'].shape[0]
    density = pcCount / layer.X / layer.Z
    if pc.planarDensity is None:
        density /= layer.Y
        densities[index, :] = [pc.density, density]
    else:
        densities[index, :] = [pc.planarDensity, density]
    index += 1
    if angle > 0.5 and show_once:
        show_once = False
        plotNetwork(scaffoldInstance, from_memory=True)

plt.plot(angle_range, densities[:, 0], label='Expected density')
plt.plot(angle_range, densities[:, 1], label='Actual density')
plt.show(block=True)
