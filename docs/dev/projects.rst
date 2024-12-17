.. _projects:

####################################
Configuring your BSB Python Projects
####################################

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
