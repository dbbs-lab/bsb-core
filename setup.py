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
    "scipy>=1.3.1",
    "scikit-learn>=0.20.3",
    "plotly>=4.1.0",
    "colour>=0.1.5",
    "errr>=1.0.0",
    "rtree>=0.9.7",
    "psutil>=5.8.0",
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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
        "dev": ["sphinx", "furo", "pre-commit", "black==20.8b1", "nrn-subprocess==1.3.4"],
        "neuron": ["dbbs_models~=2.0.0", "nrn-patch~=3.0.1"],
        "mpi": ["mpi4py"],
    },
)
