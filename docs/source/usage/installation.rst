=================
Istallation Guide
=================

The scaffold framework can be installed using Pip for Python 3

  .. code-block:: bash

    pip3 install dbbs-scaffold[with-plotting]

If you are not interested in using the scaffold to plot any results in that environment you can leave off ``[with-plotting]``.
This optional dependency installs Plotly.py

You can verify that the installation works with

  .. code-block:: bash

    scaffold -c=quickstart compile

This should generate an HDF5 file in your current directory. If everything looks fine
you are ready to 
