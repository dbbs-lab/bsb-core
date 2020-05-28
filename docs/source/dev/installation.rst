######################
Developer Installation
######################

To install::

  git clone git@github.com:Helveg/cerbellum-scaffold
  cd cerebellum-scaffold
  pip install -e .[dev]
  pre-commit install


Test your install with::

  python -m unittest discover -s tests
