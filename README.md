[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/bsb/badge/?version=latest)](https://bsb.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/dbbs-lab/bsb-core.svg?branch=main)](https://travis-ci.com/dbbs-lab/bsb-core)
[![codecov](https://codecov.io/gh/dbbs-lab/bsb-core/branch/main/graph/badge.svg)](https://codecov.io/gh/dbbs-lab/bsb-core)

<h3>:closed_book: Read the documentation on https://bsb.readthedocs.io/en/latest</h3>

# BSB: A component framework for neural modelling

Developed by the Department of Brain and Behavioral Sciences at the University of Pavia,
the BSB is a component framework for neural modelling, which focusses on component
declarations to piece together a model. The component declarations can be made in any
supported configuration language, or using the library functions in Python. It offers
parallel reconstruction and simulation of any network topology, placement and/or
connectivity strategy.


## Installation

The BSB requires Python 3.9+.

### pip

Any package in the BSB ecosystem can be installed from PyPI through `pip`. Most users
will want to install the main [bsb](https://pypi.org/project/bsb/) framework:

```
pip install "bsb~=4.1"
```

Advanced users looking to control install an unconventional combination of plugins might
be better of installing just this package, and the desired plugins:

```
pip install "bsb-core~=4.1"
```

Note that installing `bsb-core` does not come with any plugins installed and the usually
available storage engines, or configuration parsers will be missing.

### Developers

Developers best use pip's *editable* install. This creates a live link between the
installed package and the local git repository:

```
 git clone git@github.com:dbbs-lab/bsb-core
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

This creates a [network file](bsb.readthedocs.io/getting-started/networks.html) and plots the network.

### Simulating a network

The default project currently contains no simulation config.

# Contributing

All contributions are very much welcome.
Take a look at the [contribution guide](CONTRIBUTING.md)

# Acknowledgements

This research has received funding from the European Union’s Horizon 2020 Framework
Program for Research and Innovation under the Specific Grant Agreement No. 945539
(Human Brain Project SGA3) and Specific Grant Agreement No. 785907 (Human Brain
Project SGA2) and from Centro Fermi project “Local Neuronal Microcircuits” to ED. We
acknowledge the use of EBRAINS platform and Fenix Infrastructure resources, which are
partially funded from the European Union’s Horizon 2020 research and innovation
programme through the ICEI project under the grant agreement No. 800858
