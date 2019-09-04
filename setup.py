import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='scaffold',
     version='2.0',
     scripts=['scaffold.sh', 'scaffold.bat'] ,
     author="Elisa, Stefano, Claudia, Robin",
     author_email="scaffold@unipv.com",
     description="A morphologically detailed scaffolding package for the scientific modelling of the cerebellum.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/Helveg/cerebellum-scaffold",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
