=================
Istallation Guide
=================

The scaffold framework can be installed using Pip for Python 3

  .. code-block:: bash

    sudo apt-get install python3-rtree
    pip3 install --index-url=https://dbbs.glia-pkg.org dbbs-scaffold

The user name is ``hack2019`` and the password is ``hackathon2019``.

You can verify that the installation works with

  .. code-block:: bash

    scaffold -v=3 compile -x=200 -z=200 -p

This should generate an HDF5 file in your current directory and open a plot of
the generated network. If everything looks fine you are ready to advance to
the next topic.
