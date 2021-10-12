#######################
Morphology repositories
#######################

Morphology repositories (MRs) are an interface of the :mod:`.storage` module and can be
supported by the :class:`.Storage.Engine` so that morphologies can be stored inside the
network storage.

Specific morphology formats won't be supported, but if you can extract the 3D points from
them, they can be used in the framework. The MR of a network is accessible as
``network.storage.morphology_repository`` and has a
:meth:`~.storage.interfaces.MorphologyRepository.save_morphology` method to store objects.
To access a morphology you can use
:meth:`~.storage.interfaces.MorphologyRepository.load_morphology` or create a preloader
that loads the meta information, you can then use its ``load`` method to load the
morphology if you need it.

.. autoclass:: .storage.interfaces.MorphologyRepository
