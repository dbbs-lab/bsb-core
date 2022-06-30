Creating networks
=================

Default network
---------------

The default configuration contains a skeleton configuration, for an HDF5 storage, without
any components in it. The file will be called something like
``scaffold_network_2022_06_29_10_10_10.hdf5``, and will be created once you construct the
:class:`~.core.Scaffold` object:

.. literalinclude:: /../examples/networks/create_default.py
   :language: python
   :lines: 2-

Network from config
-------------------

You can also first load or create a :class:`.config.Configuration` object, and create a
network from it, by passing it to the :class:`~.core.Scaffold`:

.. literalinclude:: /../examples/networks/create_from_cfg.py
   :language: python
   :lines: 2-
