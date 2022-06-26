.. _getting-started-morpho:

Starting with morphologies
==========================

This guide is a continuation of the :doc:`Getting Started guide <getting-started>`.

We've constructed a stacked double layer topology, and we have 2 cell types. We then
connected them in an all-to-all fashion. A logical next step would be to assign
morphologies to our cells, and connect them based on intersection!

A new model never contains any morphologies, and needs to fetch them from somewhere.
Projects are configured to fetch from a local file called ``morphologies.hdf5``. Any
morphologies you place in that file will be included in your model. An alternative to
``morphologies.hdf5`` is to fetch from different sources, like NeuroMorpho. We'll go over
the different approaches.

Fetching from NeuroMorpho
-------------------------

The framework can fetch morphologies for you from `neuromorpho.org
<https://neuromorpho.org>`_. All you have to do is add a :guilabel:`morphologies` list to
your cell type:

.. tab-set-code::

  .. literalinclude:: getting-started-morpho.json
    :language: json
    :lines: 44-58
    :emphasize-lines: 7-12

  .. literalinclude:: getting_started_morpho.py
    :language: python
    :lines: 22-38

Each item in the :guilabel:`morphologies` list will be cast into a
:class:`~.morphologies.selector.MorphologySelector`. If there are multiple selectors,
their selections will be added together. Here we use the
:class:`~.morphologies.selector.NeuroMorphoSelector`. The :guilabel:`names` will be
dowmloaded from NeuroMorpho. You can find the :guilabel:`names` on the neuron info pages:

.. figure:: /images/nm_what.png
  :figwidth: 450px
  :align: center

Once you initialize your model, the framework will connect to NeuroMorpho, and download
the morphology files for you. They will be stored in your storage object, and accessible
through the ``scaffold.morphologies`` property, and the cell type's
:meth:`~.cell_types.CellType.get_morphologies` method:

.. code-block:: python

  from bsb.core import Scaffold
  from bsb.config import from_json

  cfg = from_json("network_configuration.json")
  network = Scaffold(cfg)
  top_type = network.cell_types.top_type
  names = (info.name for info in network.morphologies.all())
  top_names = (info.name for info in top_type.get_morphologies())
  print("Morphologies:", ", ".join(names))
  print("Top type morphologies:", ", ".join(names))

.. note::

	Usually when you request morphologies, you'll be handed :class:`StoredMorphologies
	<.storage.interfaces.StoredMorphology>`. They contain only the morphology metadata. If
	you want to load the morphology itself, call the
	:meth:`.storage.interfaces.StoredMorphology.load` method on them.

Fetching from the local repository
----------------------------------

The default settings of a project contain a link to ``morphologies.hdf5``. The
morphologies in it will be transferred into your network reconstructions. You can disable
this by removing the link from ``pyproject.toml`` when fetching from other sources. You
can import morphologies into this template repository by importing local files, or
constructing and saving your own :class:`~.morphologies.Morphology` objects:

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

	Download a morphology from NeuroMorpho and save it as ``my_neuron.swc`` to follow along.

Afterwards, you can use the :class:`~.morphologies.selector.NameSelector` to select your
morphologies:

.. tab-set-code::

  .. literalinclude:: getting-started-morpho.json
    :language: json
    :lines: 31-43

  .. literalinclude:: getting_started_morpho.py
    :language: python
    :lines: 17-21

Morphology intersection
-----------------------

Now that our cell types are assigned morphologies we can use some connection strategies
that use morphologies, such as
:class:`~.connectivity.detailed.voxel_intersection.VoxelIntersection`:

.. tab-set-code::

  .. literalinclude:: getting-started-morpho.json
    :language: json
    :lines: 73-83

  .. literalinclude:: getting_started_morpho.py
    :language: python
    :lines: 45-50

.. note::

  If there's multiple morphologies per cell type, they'll be assigned randomly, unless you
  specify a :class:`~.placement.strategy.MorphologyDistributor`.


Recap
-----

.. tab-set-code::

  .. literalinclude:: getting-started-morpho.json
    :language: json

  .. literalinclude:: getting_started_morpho.py
    :language: python
