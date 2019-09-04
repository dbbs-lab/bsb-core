import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='dbbs-scaffold',
     version='2.1',
     author="Elisa Marenzi, Stefano Casali, Claudia Casselato, Robin De Schepper",
     author_email="robingilbert.deschepper@unipv.it",
     description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
     scripts=['bin/scaffold', 'bin/scaffold.bat'] ,
     include_package_data=True,
     data_files=[
        ('configurations', ['scaffold/configurations/mouse_cerebellum.ini'])
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
 )
