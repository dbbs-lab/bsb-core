# Scaffold: A scaffold model for the cerebellum
This package is intended to facilitate spatially, topologically and morphologically detailed simulations of the cerebellum.

## Installation

### Pip

placeholder

```
 pip install dbbs-lab/scaffold
```

## Usage

The scaffold model can be used through the command line interface or as a python package.

### Command line interface (CLI)

1. Navigate to the directory where the scaffold package is installed.
```
cd <directory>
```

2. Run the scaffold with subcommand `compile` to compile a network architecture
```
scaffold --config=mouse_cerebellum.ini compile -p
```

To run with different configurations, change the config argument to the relative path of a .ini config file. The `-p` flag indicates that the compiled network should be plotted afterwards and can be omitted.

### Python package

The central object is the `scaffold.Scaffold` class. This object requires a `scaffold.config.ScaffoldConfig` instance for its construction. To emulate the CLI functionality you can use the `ScaffoldIniConfig` class and provide the relative path to the configuration file.

```python
from scaffold import Scaffold
from scaffold.config import ScaffoldIniConfig

config = new ScaffoldIniConfig('mouse_cerebellum.ini')
scaffoldInstance = new Scaffold(config)
```

This scaffold instance can then be used to perform the subcommands available in the CLI by calling their corresponding functions:

```python
scaffoldInstance.compileNetworkArchitecture()
```

#### Plotting network architecture

After calling `compileNetworkArchitecture` the scaffold instance can be passed to `plotNetwork` from the `scaffold.plotting` module for plotting:

```python
from scaffold.plotting import plotNetwork

plotNetwork(scaffoldInstance)
```
