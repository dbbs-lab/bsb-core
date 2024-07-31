######################
Developer Installation
######################

To install::

  git clone git@github.com:dbbs-lab/bsb-core
  cd bsb-core
  pip install -e .[dev]
  pre-commit install


Test your install with::

  cd tests/
  python -m unittest discover -s ./

