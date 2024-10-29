Accessing network data
======================

Configuration
-------------

The Configuration of a Scaffold is available as ``scaffold.configuration``.
The root components such as ``cell_types``, ``placement`` and others are available on
``scaffold`` as well.

.. literalinclude:: /../examples/networks/access_config.py
   :language: python
   :lines: 2-

Placement data
--------------

The placement data is available through the :class:`.storage.interfaces.PlacementSet`
interface. This example shows how to access the cell positions of each population:

.. literalinclude:: /../examples/networks/access_placement.py
   :language: python
   :lines: 2-

.. seealso::

   :meth:`~.storage.interfaces.PlacementSet.load_morphologies`,
   :meth:`~.storage.interfaces.PlacementSet.load_rotations`.

.. todo::

   Document best practices for the morphology data

.. todo::

   Document best practices for the connectivity data
