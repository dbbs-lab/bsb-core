import setuptools, sys, os

with open("README.md", "r") as fh:
    long_description = fh.read()


if sys.argv[1] == "develop":
    os.system("pre-commit install")

print("eeeee")
print("eeeee")
print("eeeee")

setuptools.setup(
    name="dbbs-scaffold",
    version="3.0.3",
    author="Robin De Schepper, Alice Geminiani, Stefano Casali, Alberto Antonietti, Claudia Casselato, Egidio D'Angelo",
    author_email="robingilbert.deschepper@unipv.it",
    description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
    include_package_data=True,
    data_files=[("configurations", ["scaffold/configurations/mouse_cerebellum.json"])],
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
        "rtree>=0.9.3",
    ],
    extras_require={"dev": ["sphinx", "pyarmor", "pre-commit"]},
)
