# Scaffold: A scaffold model for the cerebellum
This package is intended to facilitate spatially, topologically and morphologically detailed simulations of the cerebellum developed by the Department of Brain and Behavioral Sciences at the University of Pavia.

## Installation

### Pip

This software can be installed as a Python package from PyPI through pip.

```
 pip install dbbs-scaffold
```

**Note:** *When using Anaconda, package dependency version errors might be thrown. Try updating your
package index in the Anaconda Navigator.*

## Usage

The scaffold model can be used through the command line interface or as a python package.

### Command line interface (CLI)

Run the scaffold in the command line with subcommand `compile` to compile a network architecture.
```
scaffold --config=mouse_cerebellum.json compile -p
```

To run with different configurations, change the config argument to the relative path of a .json config file. The `-p` flag indicates that the compiled network should be plotted afterwards and can be omitted.

### Python package

The central object is the `scaffold.Scaffold` class. This object requires a `scaffold.config.ScaffoldConfig` instance for its construction. To emulate the CLI functionality you can use the `JSONConfig` class and provide the relative path to the configuration file.

```python
from scaffold import Scaffold
from scaffold.config import JSONConfig

config = new JSONConfig(file='mouse_cerebellum.json')
scaffoldInstance = new Scaffold(config)
```

This scaffold instance can then be used to perform the subcommands available in the CLI by calling their corresponding functions:

```python
scaffoldInstance.compile_network()
```

#### Plotting network architecture

After calling `compile_network` the scaffold instance can be passed to `plotNetwork` from the `scaffold.plotting` module for plotting:

```python
from scaffold.plotting import plotNetwork

plotNetwork(scaffoldInstance)
```
