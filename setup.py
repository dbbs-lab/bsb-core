import setuptools, sys, os

with open(os.path.join(os.path.dirname(__file__), "bsb", "__init__.py"), "r") as f:
    for line in f:
        if "__version__ = " in line:
            exec(line.strip())
            break

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    "h5py~=3.0",
    "numpy~=1.19",
    "scipy~=1.5",
    "scikit-learn~=1.0",
    "plotly~=5.5",
    "colour~=0.1",
    "errr~=1.0",
    "rtree~=0.9",
    "psutil~=5.8",
    "pynrrd~=0.4",
    "mpilock~=1.1",
    "morphio~=3.3",
    "mpi4py",
    "zwembad",
    "toml",
    "requests",
    "appdirs~=1.4",
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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["bsb = bsb.cli:handle_cli"],
        "bsb.adapters": [
            "arbor = bsb.simulators.arbor",
            "nest = bsb.simulators.nest",
            "neuron = bsb.simulators.neuron",
        ],
        "bsb.commands": [
            "commands = bsb.cli.commands._commands",
        ],
        "bsb.config.parsers": ["json = bsb.config.parsers.json"],
        "bsb.config.templates": ["bsb_templates = bsb.config.templates"],
        "bsb.engines": ["hdf5 = bsb.storage.engines.hdf5"],
        "bsb.options": [
            "verbosity = bsb._options:verbosity",
            "sudo = bsb._options:sudo",
            "version = bsb._options:version",
            "config = bsb._options:config",
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
            "sphinx",
            "furo",
            "pre-commit",
            "black==22.3.0",
            "nrn-subprocess==1.3.4",
            "sphinxemoji",
            "sphinx_design",
        ],
        "arbor": ["arbor~=0.6"],
        "neuron": ["dbbs_models~=2.0.0", "nrn-patch~=3.0.1"],
        "mpi": ["mpi4py", "zwembad", "mpilock"],
    },
)
