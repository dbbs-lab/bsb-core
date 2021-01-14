######################
Developer Installation
######################

To install::

  git clone git@github.com:dbbs-lab/bsb
  cd bsb
  pip install -e .[dev]
  pre-commit install


Test your install with::

  python -m unittest discover -s tests
