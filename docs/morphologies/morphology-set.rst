=============
MorphologySet
=============

.. _soft-caching:
.. _hard-caching:

Soft caching
============

Every time a morphology is loaded, it has to be read from disk and pieced together. If you
use soft caching, upon loading a morphology it is kept in cache and each time it is
re-used a copy of the cached morphology is created. This means that the storage only has
to be read once per morphology, but additional memory is used for each unique morphology
in the set. If you're iterating, the soft cache is cleared immediately after the iteration
stops. Soft caching is available by passing ``cache=True`` to
:meth:`~.morphologies.MorphologySet.iter_morphologies`:

.. code-block:: python

  from bsb.core import from_storage

  network = from_storage
  ps = network.get_placement_set("my_cell")
  ms = ps.load_morphologies()
  for morpho in ms.iter_morphologies(cache=True):
    morpho.close_gaps()
