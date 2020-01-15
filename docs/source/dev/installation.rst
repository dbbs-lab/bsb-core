######################
Developer Installation
######################

To install::

  git clone git@github.com:Helveg/cerbellum-scaffold
  cd cerebellum-scaffold
  pip3 install -e .[dev]
  pre-commit install


Test your install with::

  python3 -m unittest discover -s tests
