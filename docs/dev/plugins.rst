.. _plugins:

#######
Plugins
#######

The BSB is extensively extendible. While most smaller things such as a new placement or
connectivity strategy can be used simply by importing or dynamic configuration, larger
components such as new storage engines, configuration parsers or simulation backends are
added into the BSB through its plugin system.

Creating a plugin
=================

The plugin system detects pip packages that define ``entry_points`` of the plugin
category. Entry points can be specified in your package's ``setup`` using the
``entry_point`` argument. See the `setuptools documentation
<https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html>`_ for a full
explanation. Here are some plugins the BSB itself registers::

  entry_points={
      "bsb.adapters": [
          "nest = bsb.simulators.nest",
          "neuron = bsb.simulators.neuron",
      ],
      "bsb.engines": ["hdf5 = bsb.storage.engines.hdf5"],
      "bsb.config.parsers": ["json = bsb.config.parsers.json"],
  }

The keys of this dictionary are the plugin category that determine where the plugin will
be used while the strings that it lists follow the ``entry_point`` syntax:

* The string before the ``=`` will be used as the plugin name.
* Dotted strings indicate the module path.
* An optional ``:`` followed by a function name can be used to specify a function in the
  module.

What exactly should be returned from each ``entry_point`` depends highly on the plugin
category but there are some general rules that will be applied to the advertised object:

* The object will be checked for a ``__plugin__`` attribute, if present it will be used instead.
* If the object is a function (strictly a function, other callables are ignored), it will
  be called and the return value will be used instead.

This means that you can specify just the module of the plugin and inside the module set
the plugin object with ``__plugin__`` or define a function ``__plugin__`` that returns it.
Or if you'd like to register multiple plugins in the same module you can explicitly
specify different functions in the different entry points.

Examples
--------

In Python:

.. code-block:: python

    # my_pkg.plugin1 module
    __plugin__ = my_plugin

.. code-block:: python

    # my_pkg.plugin2 module
    def __plugin__():
        return my_awesome_adapter

.. code-block:: python

    # my_pkg.plugins
    def parser_plugin():
        return my_parser

    def storage_plugin():
        return my_storage

In ``setup``:

.. code-block:: python

    {
        "bsb.adapters": ["awesome_sim = my_pkg.plugin2"],
        "bsb.config.parsers": [
            "plugin1 = my_pkg.plugin1",
            "parser = my_pkg.plugins:parser_plugin"
        ],
        "bsb.engines": ["my_pkg.plugins:storage_plugin"]
    }

Categories
==========

Configuration parsers
---------------------

**Category:** ``bsb.config.parsers``

Inherit from :class:`.config.parsers.Parser`. When installed a ``from_<plugin-name>``
parser function is added to the :mod:`bsb.config` module. You can set the class variable
``data_description`` to describe what kind of data this parser parses to users. You can
also set ``data_extensions`` to a sequence of extensions that this parser will be
considered first for when parsing files of unknown content.


Storage engines
---------------

**Category:** ``bsb.engines``

Simulator adapters
------------------

**Category:** ``bsb.adapters``
