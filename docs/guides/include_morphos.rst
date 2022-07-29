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
Projects are configured to fetch from a local file called ``morphologies.hdf5``. Any
morphologies you place in that file will be included in your model. An alternative to
``morphologies.hdf5`` is to fetch from different sources, like NeuroMorpho. We'll go over
the different approaches.

Fetching from NeuroMorpho
-------------------------

The framework can fetch morphologies for you from `neuromorpho.org
<https://neuromorpho.org>`_. Add a :guilabel:`morphologies` list to
your ``top_type``:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 44-58
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

Fetching from the local repository
----------------------------------

By default each model in a project will fetch from ``morphologies.hdf5`` (check your
``pyproject.toml``). You can import morphologies into this template repository by
importing local files, or constructing your own :class:`~.morphologies.Morphology`
objects, and saving them:

.. code-block:: python

  from bsb.storage import Storage
  from bsb.morphologies import Morphology, Branch

  morphologies = Storage("hdf5", "morphologies.hdf5").morphologies
  # From file
  morpho = Morphology.from_swc("my_neuron.swc")
  morphologies.save("my_neuron", morpho)
  # From objects
  obj = Morphology([Branch([[0, 0, 0], [1, 1, 1]], [1])])
  morphologies.save("my_obj", obj)

.. hint::

	Download a morphology from NeuroMorpho and save it as ``my_neuron.swc`` locally.

Afterwards, we add a :class:`~.morphologies.selector.NameSelector` to the ``base_type``:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 31-43
    :emphasize-lines: 5-11

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 17-21

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
