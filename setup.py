import setuptools, sys, os

with open(os.path.join(os.path.dirname(__file__), "bsb", "__init__.py"), "r") as f:
    for line in f:
        if "__version__ = " in line:
            exec(line.strip())
            break

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    "h5py>=2.9.0",
    "numpy>=1.19.0",
    "scipy>=1.5.2",
    "scikit-learn>=0.23.2",
    "plotly>=4.1.0",
    "colour>=0.1.5",
    "errr>=1.0.0",
    "rtree>=0.9.7",
    "filelock>=3.0.0",
]

setuptools.setup(
    name="bsb",
    version=__version__,
    author="Robin De Schepper, Alice Geminiani, Alberto Antonietti, Stefano Casali, Egidio D'Angelo, Claudia Casellato",
    author_email="robingilbert.deschepper@unipv.it",
    description="A package for modelling morphologically detailed neuronal microcircuits.",
    include_package_data=True,
    package_data={"bsb": ["config/templates/*.json"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbbs-lab/bsb",
    license="GPLv3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["bsb = bsb.cli:handle_cli"],
        "bsb.adapters": [
            "nest = bsb.simulators.nest",
            "neuron = bsb.simulators.neuron",
        ],
        "bsb.commands": [
            "compile = bsb.cli.commands._commands:compile",
            "simulate = bsb.cli.commands._commands:simulate",
        ],
        "bsb.config.parsers": ["json = bsb.config.parsers.json"],
        "bsb.config.templates": ["bsb_templates = bsb.config.templates"],
        "bsb.engines": ["hdf5 = bsb.storage.engines.hdf5"],
        "bsb.options": [
            "verbosity = bsb._options:verbosity",
            "version = bsb._options:version",
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
        "dev": ["coverage", "sphinx", "furo", "pre-commit", "black==20.8b1"],
        "NEURON": ["dbbs_models>=1.3.2", "nrn-patch>=3.0.0b0"],
        "MPI": ["mpi4py", "zwembad>=1.0.0"],
    },
)
