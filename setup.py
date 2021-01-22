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
    "numpy>=1.16.4",
    "scipy>=1.3.1",
    "scikit-learn>=0.20.3",
    "plotly>=4.1.0",
    "colour>=0.1.5",
    "errr>=1.0.0",
    "rtree>=0.9.7",
]

setuptools.setup(
    name="bsb",
    version=__version__,
    author="Robin De Schepper, Alice Geminiani, Alberto Antonietti, Stefano Casali, Egidio D'Angelo, Claudia Casellato",
    author_email="robingilbert.deschepper@unipv.it",
    description="A package for modelling morphologically detailed neuronal microcircuits.",
    include_package_data=True,
    package_data={"bsb": ["configurations/*.json"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbbs-lab/bsb",
    license="GPLv3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["bsb = bsb.cli:scaffold_cli"]},
    install_requires=requires,
    project_urls={
        "Bug Tracker": "https://github.com/dbbs-lab/bsb/issues/",
        "Documentation": "https://dbbs-docs.rf.gd/",
        "Source Code": "https://github.com/dbbs-lab/bsb/",
    },
    extras_require={
        "dev": ["sphinx", "furo", "pre-commit", "black==20.8b1"],
        "neuron": ["NEURON>=7.8.1.1", "dbbs_models>=1.4.0", "nrn-patch>=3.0.0b0"],
        "mpi": ["mpi4py"],
    },
)
