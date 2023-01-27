import os

import setuptools

_findver = "__version__ = "
_rootpath = os.path.join(os.path.dirname(__file__), "bsb", "__init__.py")
with open(_rootpath, "r") as f:
    for line in f:
        if "__version__ = " in line:
            f = line.find(_findver)
            __version__ = eval(line[line.find(_findver) + len(_findver) :])
            break
    else:
        raise Exception(f"No `__version__` found in '{_rootpath}'.")

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    "bsb-hdf5~=0.6.2",
    "h5py~=3.0",
    "numpy~=1.19",
    "scipy~=1.5",
    "scikit-learn~=1.0",
    "plotly~=5.5",
    "colour~=0.1",
    "errr~=1.2",
    "rtree~=1.0",
    "psutil~=5.8",
    "pynrrd~=1.0",
    "morphio~=3.3",
    "toml",
    "requests",
    "appdirs~=1.4",
    "neo[nixio]",
    "tqdm~=4.50",
]

setuptools.setup(
    name="bsb",
    version=__version__,
    author="Robin De Schepper, Alice Geminiani, Alberto Antonietti, Stefano Casali,"
    + " Egidio D'Angelo, Claudia Casellato",
    author_email="robingilbert.deschepper@unipv.it",
    description="A component framework for modelling morphologically detailed neuronal"
    + " microcircuits",
    include_package_data=True,
    package_data={
        "bsb": [
            "config/templates/*.json",
            "unittest/data/configs/*",
            "unittest/data/morphologies/*",
            "unittest/data/parser_tests/*",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbbs-lab/bsb",
    license="GPLv3",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["bsb = bsb.cli:handle_cli"],
        "bsb.storage.engines": ["fs = bsb.storage.fs"],
        "bsb.simulation_backends": [
            "arbor = bsb.simulators.arbor",
            "nest = bsb.simulators.nest",
            "neuron = bsb.simulators.neuron",
        ],
        "bsb.commands": [
            "commands = bsb.cli.commands._commands",
            "projects = bsb.cli.commands._projects",
        ],
        "bsb.config.parsers": ["json = bsb.config.parsers.json"],
        "bsb.config.templates": ["bsb_templates = bsb.config.templates"],
        "bsb.options": [
            "verbosity = bsb._options:verbosity",
            "sudo = bsb._options:sudo",
            "version = bsb._options:version",
            "config = bsb._options:config",
            "profiling = bsb._options:profiling",
        ],
    },
    python_requires="~=3.8",
    install_requires=requires,
    project_urls={
        "Bug Tracker": "https://github.com/dbbs-lab/bsb/issues/",
        "Documentation": "https://bsb.readthedocs.io/",
        "Source Code": "https://github.com/dbbs-lab/bsb/",
    },
    extras_require={
        "dev": [
            "sphinx~=5.0",
            "furo",
            "pre-commit",
            "black~=22.3.0",
            "nrn-subprocess~=1.3.4",
            "sphinxemoji",
            "sphinx_design~=0.2",
            "sphinx-copybutton~=0.5",
            "sphinxext-bsb~=0.0.2",
            "snakeviz",
        ],
        "arbor": ["arbor~=0.6", "arborize[arbor]==4.0.0a4"],
        "neuron": ["nrn-patch==4.0.0a4", "arborize[neuron]==4.0.0a4"],
        "mpi": ["mpi4py~=3.0", "zwembad", "mpilock~=1.1"],
    },
)
