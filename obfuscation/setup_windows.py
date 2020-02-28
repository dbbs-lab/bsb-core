import setuptools, os, sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import scaffold

with open("../README.md", "r") as fh:
    long_description = fh.read()

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False


except ImportError:
    bdist_wheel = None

setuptools.setup(
    name="dbbs-scaffold",
    version=scaffold.__version__,
    author="Robin De Schepper, Alice Geminiani, Alberto Antonietti, Stefano Casali, Claudia Casselato, Egidio D'Angelo",
    author_email="robingilbert.deschepper@unipv.it",
    description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
    include_package_data=True,
    package_data={"scaffold": ["configurations/*.json", "pytransform/*"]},
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
    extras_require={"dev": ["sphinx", "pyarmor"]},
    cmdclass={"bdist_wheel": bdist_wheel},
)
