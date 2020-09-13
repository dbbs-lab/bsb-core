[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/bsb/badge/?version=latest)](https://bsb.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/dbbs-lab/bsb.svg?branch=master)](https://travis-ci.com/dbbs-lab/bsb)
[![codecov](https://codecov.io/gh/dbbs-lab/bsb/branch/master/graph/badge.svg)](https://codecov.io/gh/dbbs-lab/bsb)

**Note:** The scaffold framework is still under heavy development. Please check the
**_Known Issues_** section at the bottom for important issues that fell victim to our
deadlines and will be solved at a later date.

# BSB: A scaffold modelling framework
This package is intended to facilitate spatially, topologically and morphologically
detailed simulations of brain regions developed by the Department of Brain and
Behavioral Sciences at the University of Pavia.

## Installation

The BSB requires Python 3.8+.

### pip

This software can be installed as a Python package from PyPI through pip:

```
pip install bsb
```

### Developers

Developers best use pip's *editable* install. This creates a live link between the
installed package and the local git repository:

```
 git clone git@github.com:dbbs-lab/bsb
 cd bsb
 pip install -e .[dev]
 pre-commit install
```

## Usage

The scaffold framework can be used through the command line interface or as a python package.

### Command line interface (CLI)

Run the framework in the command line with subcommand `compile` to compile a network
architecture.

```
bsb --config=mouse_cerebellum_cortex_noTouch.json compile -x=200 -z=200 -p
```

To run with different configurations, change the config argument to the relative path of a
.json config file. The `-p` flag indicates that the compiled network should be plotted
afterwards and can be omitted.

### Python package

The central object is the `bsb.core.Scaffold` class. This object requires a
`bsb.config.ScaffoldConfig` instance for its construction. To emulate the CLI
functionality you can use the `JSONConfig` class and provide the relative path to the
configuration file.

```python
from bsb import Scaffold
from bsb.config import JSONConfig

config = new JSONConfig(file='mouse_cerebellum_cortex_noTouch.json')
scaffoldInstance = new Scaffold(config)
```

This scaffold instance can then be used to perform the subcommands available in the CLI by
calling their corresponding functions:

```python
scaffoldInstance.compile_network()
```

#### Plotting network architecture

After calling `compile_network` the scaffold instance can be plotted:

```python
scaffoldInstance.plot_network_cache()
```


# Known issues

## No configuration serialization

When modifying the config object through scripts and then saving it to file, you'll store
the original configuration file text, and you won't actually serialize the modified object

We will fix this by version 4.0

## If MPI is used but mpi4py is not installed undefined behavior may occur

The BSB determines the amount of NEST virtual processes by using mpi4py to get the amount
of MPI processes. But if the package is not installed it is assumed no MPI simulations
will be ran and the amount of virtual processes might be lower than expected when used in
combination with OpenMP. Be sure to `pip install` using the `[MPI]` requirement tag.
