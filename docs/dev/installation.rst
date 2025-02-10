######################
Developer Installation
######################

.. start-dev-install

To install bsb from the source code:

.. code-block:: bash

  git clone git@github.com:dbbs-lab/bsb-core
  cd bsb-core
  pip install -e .[dev]
  pre-commit install

The ``dev`` installation contains all the necessary libraries to run bsb, perform the unittests and build the
bsb documentation. You can test your installation with:

.. code-block:: bash

  cd tests/
  python -m unittest discover -s ./

