import numpy as np
from bsb.core import from_hdf5
from bsb.storage import Storage
from bsb.config import from_json
from bsb.core import Scaffold
from bsb.morphologies import Morphology
import h5py
from bsb.plotting import plot_morphology
from bsb.plotting import get_branch_trace
import plotly.graph_objects as go
#cell = Morphology.from_swc("morphologies/PyramidalCell.swc")

####or
net = Storage("hdf5", "VISp.hdf5")
mr = net.morphologies
cell = mr.load("test1")
# print(np.sum(cell.tags == 1))
# print(np.sum(cell.tags == 2))
# print(np.sum(cell.tags == 3))
# print(np.sum(cell.tags == 4))

print("numb of branches on the morphology: ", len(cell.branches))
print("numb of points on the morphology: ", len(cell.tags))
print(cell.labelsets)
#plot_morphology(cell)
# print(cell.tags)
#
#
# branches = [b for b in cell.branches if np.all(b.tags == 3)]
# for b in branches:
#     b.label(["basal_dendrites"])
# branches = [b for b in cell.branches if np.all(b.tags == 4)]
# for b in branches:
#     b.label(["apical_dendrites"])
#### to add specific labels on specific points
#cell.label(["my_label"], [15, 30, 45])

# dendrites = cell.labels.get_mask(["dendrites"])
# dendrites
# apical_dendrites = cell.get_branches(labels=["apical_dendrites"])
# # # np.all(dendrites == (pc.tags == 3))
# for b in apical_dendrites:
#     b.label(["dendrites"])
# #cell.labels.label(["dendrites", "labels"], apical_dendrites)
# mr.save("test2", cell)

####### PLOTTING
# rng = cell.bounds
# print(type(rng))
# print(rng)
# rng_absMin = np.min(cell.bounds[0])
# rng_absMax = np.max(cell.bounds[1])
# rngL=list(rng)
# rngL[0] = np.array([rng_absMin,rng_absMin, rng_absMin])
# rngL[1] = np.array([rng_absMax,rng_absMax, rng_absMax])
# rng=tuple(rngL)
# print(rng)
# print(type(rng))
plot_morphology(cell, color={'soma': 'black', 'apical_dendrites': 'green', 'axon': 'violet', 'basal_dendrites': 'blue'})


# fig = go.Figure()
# color="black"
# width=1.0
# traces = []
# for branch in cell.branches:
#     print(branch.label)
#     traces.append(get_branch_trace(branch, offset=None, color=color, width=width))
# for trace in traces:
#     fig.add_trace(trace)
# fig.show()
