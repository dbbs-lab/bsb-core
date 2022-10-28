#######################
Morphology repositories
#######################

Morphology repositories (MRs) are an interface of the :mod:`.storage` module and can be
supported by the :class:`~.storage.interfaces.Engine` so that morphologies can be stored
inside the network storage.

To access an MR, a :class:`~.storage.Storage` object is required:

.. code-block:: python

  from bsb.storage import Storage

  store = Storage("hdf5", "morphologies.hdf5")
  mr = store.morphologies
  print(mr.all())

Similarly, the built-in MR of a network is accessible as ``network.morphologies``:

.. code-block:: python

  from bsb.core import from_storage

  network = from_hdf("my_existing_model.hdf5")
  mr = network.morphologies

You can use the :meth:`~.storage.interfaces.MorphologyRepository.save` method to store
:class:`Morphologies <.morphologies.Morphology>`. If you don't immediately need the whole
morphology, you can :meth:`~.storage.interfaces.MorphologyRepository.preload` it,
otherwise you can load the entire thing with
:meth:`~.storage.interfaces.MorphologyRepository.load`.

.. autoclass:: bsb.storage.interfaces.MorphologyRepository
  :noindex:
  :members:
