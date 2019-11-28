#############
Configuration
#############

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  configuration/cell-type
  configuration/connection-type
  configuration/layer
  configuration/placement-strategies
  configuration/connection-strategies

.. note::
  The key of a configuration object in its parent will be stored as its ``name``
  property and is used throughout the package. Some of these values are
  hardcoded into the package and the names of the standard configuration objects
  should not be changed.

==========
Attributes
==========

The root node accepts the following attributes:

* ``name``: *Unused*, a name for the configuration file. Is stored in the output
  files so it can be used for reference.
* ``output``: Configuration object for the output :class:`HDF5Formatter`.
* ``network_architecture``: Configuration object for general simulation properties.
* ``layers``: A dictionary containing the :class:`Layer` configurations.
* ``cell_types``: A dictionary containing the :class:`CellType` configurations.
* ``connection_types``: A dictionary containing the :class:`ConnectionStrategy` configurations.
* ``simulations``: A dictionary containing the :class:`SimulationAdapter` configurations.

Output attributes
=================

Format
------

This attribute is a string that refers to the implementation of the OutputFormatter
that should be used::

  {
    "output": {
      "format": "scaffold.output.HDF5Formatter"
    }
  }

If you write your own implementation the string should be discoverable by Python.
Here is an example for ``MyOutputFormatter`` in a package called ``my_package``::

  {
    "output": {
      "format": "my_package.MyOutputFormatter"
    }
  }

File
----

Determines the path and filename of the output file produced by the output
formatter. This path is relative to Python's current working directory.

  {
    "output": {
      "file": "my_file.hdf5"
    }
  }
