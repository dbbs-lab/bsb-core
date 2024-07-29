######################
Developer Installation
######################

To install::

  git clone git@github.com:dbbs-lab/bsb-core
  cd bsb-core
  pip install -e .[dev]
  cd tests/
  pre-commit install


Test your install with::

  python -m unittest discover -s tests

