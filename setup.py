import setuptools, sys, os

with open(os.path.join(os.path.dirname(__file__), "bsb", "__init__.py"), "r") as f:
    for line in f:
        if "__version__ = " in line:
            exec(line.strip())
            break

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    install_requires=[
        "h5py>=2.9.0",
        "numpy>=1.16.4",
        "scipy>=1.3.1",
        "scikit-learn>=0.20.3",
        "plotly>=4.1.0",
        "rtree-linux==0.9.4",
        "nrn-patch>=2.1.0",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/dbbs-lab/bsb/issues/",
        "Documentation": "https://dbbs-docs.rf.gd/",
        "Source Code": "https://github.com/dbbs-lab/bsb/",
    },
    extras_require={
        "dev": ["sphinx", "sphinx_rtd_theme>=0.4.3", "pyarmor", "pre-commit", "black"],
        "NEURON": ["dbbs_models>=0.4.4"],
    },
)
