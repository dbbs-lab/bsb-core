######################
Developer Installation
######################

To install::

  git clone git@github.com:dbbs-lab/bsb-core
  cd bsb-core
  pip install -e .[dev]
  pre-commit install -t pre-commit -t commit-msg


Test your install with::

  python -m unittest discover -s tests

Releases
--------

To release a new version::

  bump-my-version bump pre_n
  python -m build
  twine upload dist/* --skip-existing