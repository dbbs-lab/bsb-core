.. _include_morphos:

Adding morphologies
===================

This guide is a continuation of the |:books:| :doc:`Getting Started guide
</usage/getting-started>`.

We've constructed a stacked double layer topology, and we have 2 cell types. We then
connected them in an all-to-all fashion. A logical next step would be to assign
:doc:`morphologies </morphologies/intro>` to our cells, and connect them based on
intersection!

A new model never contains any morphologies, and needs to fetch them from somewhere.
It is possible to load a local file or to fetch from different sources, like NeuroMorpho.

Fetching from the local repository
----------------------------------
The BSB Scaffold contains a Morphology Repository object where is possible to store
morphology templates. You can import morphologies into this template repository by
importing local files, or constructing your own :class:`~.morphologies.Morphology`
objects, and saving them:

.. tab-set-code::
  .. code-block:: python

   from bsb.core import Scaffold
   from bsb.config import from_json
   import bsb.options
   from bsb.morphologies import Morphology

   bsb.options.verbosity = 3

   morpho = Morphology.from_swc("my_neuron.swc")

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 11-17

.. hint::

	Download a morphology from NeuroMorpho and save it as ``my_neuron.swc`` locally.

In this case a Morphology is created from "my_neuron.swc" with the name "my_neuron".
Afterwards, we add a :class:`~.morphologies.selector.NameSelector` to the ``base_type``:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 36-48
    :emphasize-lines: 6-12

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 18-22


Fetching from NeuroMorpho
-------------------------

The framework can fetch morphologies for you from `neuromorpho.org
<https://neuromorpho.org>`_. Add a :guilabel:`morphologies` list to
your ``top_type``:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 51-65
    :emphasize-lines: 5-14

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 22-38

.. tip::

	The :guilabel:`morphologies` attribute is a **list**. Each item in the list is a
	:class:`selector <.morphologies.selector.MorphologySelector>`. Each selector selects a
	set of morphologies from the repository, and those selections are added together and
	assigned to the population.

Each item in the :guilabel:`names` attribute will be downloaded from NeuroMorpho. You can
find the :guilabel:`names` on the neuron info pages:

.. figure:: /images/nm_what.png
  :figwidth: 450px
  :align: center

.. Once you initialize your model, the framework will connect to NeuroMorpho, and download
.. the morphology files for you. They will be stored in your storage object, and accessible
.. through the ``scaffold.morphologies`` property, and the cell type's
.. :meth:`~.cell_types.CellType.get_morphologies` method:
..
.. .. code-block:: python
..
..   from bsb.core import Scaffold
..   from bsb.config import from_json
..
..   cfg = from_json("network_configuration.json")
..   network = Scaffold(cfg)
..   top_type = network.cell_types.top_type
..   names = (info.name for info in network.morphologies.all())
..   top_names = (info.name for info in top_type.get_morphologies())
..   print("Morphologies:", ", ".join(names))
..   print("Top type morphologies:", ", ".join(names))
..
.. .. note::
..
.. 	Usually when you request morphologies, you'll be handed :class:`StoredMorphologies
.. 	<.storage.interfaces.StoredMorphology>`. They contain only the morphology metadata. If
.. 	you want to load the morphology itself, call the
.. 	:meth:`.storage.interfaces.StoredMorphology.load` method on them.



Morphology intersection
-----------------------

Now that our cell types are assigned morphologies we can use some connection strategies
that use morphologies, such as
:class:`~.connectivity.detailed.voxel_intersection.VoxelIntersection`:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 73-83

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 45-50

.. note::

  If there's multiple morphologies per cell type, they'll be assigned randomly, unless you
  specify a :class:`~.placement.distributor.MorphologyDistributor`.


Recap
-----

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json

  .. literalinclude:: include_morphos.py
    :language: python
