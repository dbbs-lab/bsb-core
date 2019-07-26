import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotNetwork(scaffold, file=None, from_memory=False):
    if from_memory:
        plt.interactive(True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for type in scaffold.configuration.cell_types.values():
            pos = scaffold.cells_by_type[type.name]
            color = type.color
            ax.scatter3D(pos[:,0], pos[:,1], pos[:,2],c=color)
            print('{} {} placed.'.format(pos.shape[0], type.name))
        plt.show(block=True)
