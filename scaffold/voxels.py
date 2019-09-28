from scaffold.helpers import dimensions, origin
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from time import sleep

class VoxelCloud:
    def __init__(self, bounds, grid_size, voxels, map):
        self.bounds = bounds
        self.grid_size = grid_size
        self.voxels = voxels
        self.map = map

    def get_boxes():
        return np.mgrid[
            self.bounds[0, 0]:self.bounds[0, 1]:grid_size,
            self.bounds[1, 0]:self.bounds[1, 1]:grid_size,
            self.bounds[2, 0]:self.bounds[2, 1]:grid_size
        ]

class Box(dimensions, origin):
    pass
