import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotNetwork(scaffold, file=None, from_memory=False):
    if from_memory:
        plt.interactive(True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for type in scaffold.configuration.CellTypes.values():
            pos = scaffold.final_cell_positions[type.name]
            color = type.color
            ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],c=color)
        plt.show(block=True)
