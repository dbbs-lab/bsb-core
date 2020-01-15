import setuptools

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
     name='dbbs-scaffold',
     version='3.0.4',
     author="Robin De Schepper, Alice Geminiani, Stefano Casali, Alberto Antonietti, Claudia Casselato, Egidio D'Angelo",
     author_email="robingilbert.deschepper@unipv.it",
     description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
     include_package_data=True,
     data_files=[
        ('configurations', ['../scaffold/configurations/mouse_cerebellum.json']),
        ('pytransform', ['scaffold/pytransform/_pytransform.so', 'scaffold/pytransform/license.lic', 'scaffold/pytransform/pytransform.key'])
     ],
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/dbbs-lab/scaffold",
     license='GPLv3',
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     entry_points={
        'console_scripts': [
            'scaffold = scaffold.cli:scaffold_cli'
        ]
     },
     install_requires= [
         'h5py>=2.9.0',
         'numpy>=1.16.4',
         'scipy>=1.3.1',
         'scikit-learn>=0.20.3',
         'plotly>=4.1.0',
         'rtree>=0.9.3'
     ],
     extras_require= {
        'dev': ['sphinx', 'pyarmor']
     },
     cmdclass={'bdist_wheel': bdist_wheel}
 )
