#############
Documentation
#############

Install the documentation dependencies of the BSB:

.. code-block:: bash

  pip install -e .[docs]

Then navigate to the ``docs`` directory and run:

.. code-block:: bash

  cd docs
  make html

The output will be in the ``/docs/_build`` folder.

Conventions
===========

* Values are marked as ``5`` or ``"hello"`` using double backticks (\`\` \`\`).
* Configuration attributes are marked as :guilabel:`attribute` using the guilabel
  directive (``:guilabel:`attribute```)
