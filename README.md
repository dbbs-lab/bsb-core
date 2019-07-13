# Unnamed: A scaffold for modelling the cerebellum
 Let's have some fun with a name. What about PyScaff, PyCereb, scaff, ... ? :) Best name gets a Belgian beer ;p
 
 GitHub Repo for Scaffold model of Cerebellum.

 Code use for the Frontiers in Neuroinformatics paper:
 https://www.frontiersin.org/articles/10.3389/fninf.2019.00037/full
 
## Installation

### Conda

placeholder

```
 conda install <our-package>
```

### Pip

placeholder

```
 pip install <our-package>
```

## Usage

Adapt `mouse_cerebellum.ini` and use

```
scaffold compile
```

or to run the nest simulation

```
scaffold run
```

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
