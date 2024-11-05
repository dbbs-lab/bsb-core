.. _projects:

####################################
Configuring your BSB Python Projects
####################################

Projects help you keep your models organized, safe, and neat! A project is a folder
containing:

* The ``pyproject.toml`` Python project settings file:
  This file uses the TOML syntax to set configuration values for the BSB and any other
  python tools your project uses.
* One or more configuration files.
* One or more network files.
* Your component code.

Remember that you can create an empty Python project for BSB using the
:ref:`bsb.new <bsb_new>` command.

Settings
========

Python project settings are contained in the ``pyproject.toml`` file.
A lot of Python options can be configured with your ``toml`` file such as the python
libraries necessary to deploy it. If you want to learn more this configuration tool,
check out `this tutorial <https://realpython.com/python-toml/>`

* ``[tools.bsb]``: The root configuration section:
  You can set the values of any :doc:`/cli/options` here.

  * ``[tools.bsb.links]``: Contains the :ref:`file link <file_link>` definitions.

  * ``[tools.bsb.links."my_network.hdf5"]``: Storage specific file links
    In this example for a storage object called "my_network.hdf5"

.. code-block:: toml

  [tools.bsb]
  verbosity = 3

  [tools.bsb.links]
  config = "auto"

  [tools.bsb.links."thalamus.hdf5"]
  config = [ "sys", "thalamus.json", "always",]

.. _file_link:

File links
==========

Remember that the `Storage` keeps copies of your `Configuration` and any data attached to it.
For instance, a copy of each unique `Morphology` attached to your cell types is stored within
your `Storage`. Now, these copies might become outdated during development.
Fortunately, you can automatically update them, using file links.

.. warning::
    It is recommended that you only specify links for models that you are actively developing,
    to avoid overwriting and losing any unique configs or morphologies of a model.

Config links
------------

Configuration links (``config =``) can be either *fixed* or *automatic*.

- Fixed config links will always overwrite the stored data of your `Scaffold` with the
  contents of the file, if it exists.
- Automatic config links do the same, but keep track of the path of the last saved config
  file, and stay linked with that file.

Syntax
------

For each file to link, you need to provide a list of 3 parameters:

- The first argument is the *provider* of the link (i.e., the engine to access the file):
  Most of the time, your will use ``sys`` for the file system (your folder). Note that you
  can also specify any of the BSB :doc:`storage engines </core/storage>` such as ``fs``.
- The second argument is the path to the file,
- the third argument is when to update, but is unused! For automatic config links you can
  simply pass the ``"auto"`` string.

.. note::

  Links in ``tools.bsb.links`` are active for all models in your project! It's better to
  specify them on a per model basis using the ``tools.bsb.links."my_model_name.hdf5"``
  section.


Component code
==============

It is best practice to keep your component code in a subfolder with the same name as
your model. For example, if you are modelling the cerebellum, create a folder called
``cerebellum``. Inside place an ``__init__.py`` file, so that Python can import code from
it. Then you best subdivide your code based on component type, e.g. keep placement
strategies in a file called ``placement.py``. That way, your placement components are
available in your model as ``cerebellum.placement.MyComponent``. It will also make it
easy to distribute your code as a package!

Version control
===============

An often overlooked aspect of a code project management is its version control!
Version control helps you track every change you make as a version of your code, backs up
your code, and lets you switch between versions.
The ``git`` protocol is currently the most popular version control, combined
with providers like `GitHub <https://github.com/>`_ or `GitLab <https://gitlab.com/>`_.

.. code-block:: diff

  - This was my previous version
  + This is my new version
  This line was not affected

This example shows how version control can track every change you make, to undo work, to
try experimental changes, or to work on multiple conflicting features. Every change can be
stored as a version, and backed up in the cloud.

.. tip::
    If it is not the case already, we highly recommend that your familiarize yourself with
    ``git`` and Github (see `this tutorial <https://github.com/git-guides>`_).

Git projects come with a ``.gitignore`` file, where you can exclude files from being backed
up. Usually, only code should be pushed online and `a contrario` large files (e.g. your
network HDF5 file) should be excluded. Please note that most cloud providers won't let
neuroscientists upload their 100GB network files |:innocent:|

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Command-Line Interface
       :link: cli-guide
       :link-type: ref

       Familiarize yourself with BSB's CLI.

    .. grid-item-card:: :octicon:`gear;1em;sd-text-warning` Learn about components
       :link: main-components
       :link-type: ref

       Explore more about the main components.

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
        :link: examples
        :link-type: ref

        Explore more advanced examples

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Make custom components
       :link: components
       :link-type: ref

       Learn how to write your own components to e.g. place or connect cells.
