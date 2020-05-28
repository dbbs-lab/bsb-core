import setuptools, sys, os
import scaffold

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dbbs-scaffold",
    version=scaffold.__version__,
    author="Robin De Schepper, Alice Geminiani, Alberto Antonietti, Stefano Casali, Claudia Casellato, Egidio D'Angelo",
    author_email="robingilbert.deschepper@unipv.it",
    description="A package for modelling morphologically detailed neuronal microcircuits.",
    include_package_data=True,
    package_data={"scaffold": ["configurations/*.json"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbbs-lab/scaffold",
    license="GPLv3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["scaffold = scaffold.cli:scaffold_cli"]},
    install_requires=[
        "h5py>=2.9.0",
        "numpy>=1.16.4",
        "scipy>=1.3.1",
        "scikit-learn>=0.20.3",
        "plotly>=4.1.0",
        "rtree-linux==0.9.4",
        "nrn-patch>=2.0.0",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/dbbs-lab/scaffold/issues/",
        "Documentation": "https://dbbs-docs.rf.gd/",
        "Source Code": "https://github.com/dbbs-lab/scaffold/",
    },
    extras_require={
        "dev": ["sphinx", "sphinx_rtd_theme>=0.4.3", "pyarmor", "pre-commit", "black"],
        "NEURON": ["dbbs_models>=0.4.3"],
    },
)
