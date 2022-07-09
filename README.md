[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/bsb/badge/?version=latest)](https://bsb.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/dbbs-lab/bsb.svg?branch=master)](https://travis-ci.com/dbbs-lab/bsb)
[![codecov](https://codecov.io/gh/dbbs-lab/bsb/branch/master/graph/badge.svg)](https://codecov.io/gh/dbbs-lab/bsb)

<h3>:closed_book: Read the documentation on https://bsb.readthedocs.io/en/latest</h3>

# BSB: A component framework for neural modelling

Developed by the Department of Brain and Behavioral Sciences at the University of Pavia,
the BSB is a component framework for neural modelling, which focusses on component
declarations to piece together a model. The component declarations can be made in any
supported configuration language, or using the library functions in Python. It offers
parallel reconstruction and simulation of any network topology, placement and/or
connectivity strategy.


## Installation

The BSB requires Python 3.8+.

### pip

This software can be installed as a Python package from PyPI through pip:

```
pip install "bsb>=4.0.0a0"
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

The scaffold framework is best used in a project context. Create a working directory for
each of your modelling projects and use the command line to configure, reconstruct or
simulate your models.

### Creating a project

You can create a quickstart project using:

```
bsb new my_model --quickstart
cd my_model
```

### Reconstructing a network

You can use your project to create reconstructions of your model, generating cell positions
and connections:

```
bsb compile -p
```

This creates a [network file](bsb.readthedocs.io/guides/networks.html) and plots the network.

### Simulating a network

The default project currently contains no simulation config.

# Contributing

All contributions are very much welcome.
Take a look at the [contribution guide](CONTRIBUTING.md)

# Known issues

## Simulation interfaces are not reinstated yet in v4

Shouldn't be much work, famous last words.
