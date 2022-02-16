#######################
Morphology repositories
#######################

Morphology repositories (MRs) are an interface of the :mod:`.storage` module and can be
supported by the :class:`~.storage.interfaces.Engine` so that morphologies can be stored
inside the network storage.

The MR of a network is accessible as ``network.morphologies`` and has a
:meth:`~.storage.interfaces.MorphologyRepository.save` method to store
:class:`~.morphologies.Morphology`. To access a :class:`~.morphologies.Morphology` you can
use :meth:`~.storage.interfaces.MorphologyRepository.load` or create a preloader that
loads the meta information, you can then use its ``load`` method to load the
:class:`~.morphologies.Morphology` if you need it.

.. autoclass:: bsb.storage.interfaces.MorphologyRepository
  :noindex:
