########
Projects
########

Projects help you keep your models organized, safe, and neat! A project is a folder
containing:

* The ``pyproject.toml`` Python project settings file:
  This file uses the TOML syntax to set configuration values for the BSB and any other
  python tools your project uses.

* One or more configuration files.

* One or more network files.

* Your component code.

You can create projects using the :ref:`bsb.new <bsb_new>` command.

Settings
========

Project settings are contained in the ``pyproject.toml`` file.

* ``[tools.bsb]``: The root configuration section:
  You can set the values of any :doc:`/cli/options` here.

  * ``[tools.bsb.links]``: Contains the :ref:`file link <file_link>` definitions.

  * ``[tools.bsb.links."my_network.hdf5"]``: Storage specific file links
    In this example for a storage object called "my_network.hdf5"

.. code-block:: toml

  [tools.bsb]
  verbosity = 3

  [tools.bsb.links]
  morpho = [ "sys", "morphologies.hdf5", "newer",]
  config = "auto"

  [tools.bsb.links."thalamus.hdf5"]
  config = [ "sys", "thalamus.json", "always",]

.. _file_link:

File links
==========

Storage objects can keep copies of configuration and morphologies. These copies might
become outdated during development. To automatically update it, you can specify file
links.

It is recommended that you only specify links for models that you are actively developing,
to avoid overwriting and losing any unique configs or morphologies of a model.

Config links
------------

Configuration links (``config =``) can be either *fixed* or *automatic*. Fixed config
links will always overwrite the configuration of the model with the contents of the file,
if it exists. Automatic config links do the same, but keep track of the path of the last
saved config file, and stay linked with that file.

Syntax
------

The first argument is the *provider* of the link: ``sys`` for the filesystem (your folder)
``fs`` for the file store of the storage engine (storage engines may have their own way of
storing files). The second argument is the path to the file, and the third argument is
when to update, but is unused! For automatic config links you can simply pass the
``"auto"`` string.

.. note::

  Links in ``tools.bsb.links`` are active for all models in your project! It's better to
  specify them on a per model basis using the ``tools.bsb.links."my_model_name.hdf5"``
  section.


Component code
==============

It's best practice to keep all of your component code in a subfolder with the same name as
your model. For example, if you're modelling the cerebellum, create a folder called
``cerebellum``. Inside place an ``__init__.py`` file, so that Python can import code from
it. Then you best subdivide your code based on component type, e.g. keep placement
strategies in a file called ``placement.py``. That way, your placement components are
available in your model as ``cerebellum.placement.MyComponent``. It will also make it
easy to distribute your code as a package!

Version control
===============

An often overlooked aspect is version control! Version control helps you track every
change you make as a version of your code, backs up your code, and lets you switch between
versions. The ``git`` protocol is currently the most popular version control, combined
with providers like GitHub or GitLab.

.. code-block:: diff

  - This was my previous version
  + This is my new version
  This line was not affected

This example shows how version control can track every change you make, to undo work, to
try experimental changes, or to work on multiple conflicting features. Every change can be
stored as a version, and backed up in the cloud.

Projects come with a ``.gitignore`` file, where you can exclude files from being backed
up. Cloud providers won't let neuroscientists upload 100GB network files |:innocent:|
