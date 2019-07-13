# scaffold_model_cerebellum
 GitHub Repo for Scaffold model of Cerebellum.

 Code use for the Frontiers in Neuroinformatics paper:
 https://www.frontiersin.org/articles/10.3389/fninf.2019.00037/full

## Plotting network architecture

```
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
f = h5py.File('YOUR_NETWORK_ARCHITECTURE_FILE.hdf5', 'r')
pos = np.array(f['positions'])
ax.scatter3D(pos[:,2], pos[:,4], pos[:,3])
```
