import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='dbbs-scaffold',
     version='2.0.1',
     scripts=['scaffold.sh', 'scaffold.bat'] ,
     author="Elisa Marenzi, Stefano Casali, Claudia Casselato, Robin De Schepper",
     author_email="robingilbert.deschepper@unipv.it",
     description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
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
