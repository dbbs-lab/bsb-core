######################
Developer Installation
######################

To install::

  git clone git@github.com:Helveg/cerbellum-scaffold
  cd cerebellum-scaffold
  sudo apt-get install python3-rtree
  pip install -e .[dev]
  pre-commit install


Test your install with::

  python -m unittest discover -s tests
