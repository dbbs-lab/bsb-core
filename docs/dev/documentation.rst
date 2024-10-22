#############
Documentation
#############

The libraries necessary for building the documentation can be installed through:

.. code-block:: bash

  pip install -e .[docs]

You should have it if you install it with the ``dev`` flag.
You can build the documentations by navigate to the ``docs`` directory and run:

.. code-block:: bash

  cd docs
  make html

The output will be in the ``/docs/_build`` folder.

.. note::
    Note that the command ``make html`` by default does not show you warnings in the documentations.
    These warnings will not pass the tests on the Github repository. To test if the documentations
    was properly implemented, prefer the command:

    .. code-block:: bash

        sphinx-build -nW -b html . _build/html

Conventions
===========

| Except for the files located at the root of the project (e.g.: README.md), the documentation is written in
  `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ . Docstrings
  in the python code should therefore be in the reStructuredText (``reST``) format.
| In the documentation, the following rules should be implemented:

* Values are marked as ``5`` or ``"hello"`` using double backticks (\`\` \`\`).
* Configuration attributes are marked as :guilabel:`attribute` using the guilabel
  directive (``:guilabel:`attribute```)
