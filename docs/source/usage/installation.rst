=================
Istallation Guide
=================

The scaffold framework can be installed using Pip for Python 3

  .. code-block:: bash

    sudo apt-get install python3-rtree
    pip3 install --index-url=https://dbbs.glia-pkg.org plotly==4.1.0 dbbs-scaffold
    export PATH=$PATH:/home/mizzou/.local/bin

The user name is ``hack2019`` and the password is ``hackathon2019``.

You can verify that the installation works with

  .. code-block:: bash

    scaffold compile -v=3 -x=200 -z=200 -p

This should generate an HDF5 file in your current directory and open a plot of
the generated network. If everything looks fine you are ready to advance to
the next topic.
