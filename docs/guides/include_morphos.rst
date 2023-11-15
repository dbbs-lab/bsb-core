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

Using local files
-----------------
The BSB Scaffold contains a Morphology Repository object where is possible to store
morphology templates. You can import morphologies into this template repository by
importing local files, or constructing your own :class:`~.morphologies.Morphology`
objects, and saving them:

.. tab-set-code::

  .. code-block:: json

       "morphologies": [
         "neuron_A.swc"
       ],

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 18

.. hint::

    Download a morphology from NeuroMorpho and save it as ``neuron_A.swc`` locally.

In this case a Morphology is created from ``neuron_A.swc`` with the name ``"neuron_A"``.
As a second step, we associate this morphology to the :guilabel:`morphologies` of our cell types by its name:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 41-50
    :emphasize-lines: 6-9

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 21


By default the name assigned to the morphology is the file name without ``.swc`` extension, to
change the name we can edit the attribute:

.. tab-set-code::

  .. code-block:: json

       "morphologies": [
         {
           "name": "neuron_B",
           "file": "my_other_neuron.swc"
         }

       ],

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 19


It is also possible to add a pipeline to perform transformations on the loaded
morphology. There is a set of implemented actions listed here :ref:`transform`
where we can select the method and assign it to the :guilabel:`pipeline` attribute with the
addition of parameters if it is required. Another option is to use user defined functions.

.. code-block:: json

  "morphologies": [
    {
      "name": "my_neuron",
      "file": "my_neuron.swc",
      "pipeline": [
        "center",
        "my_module.add_axon",
        {
          "func": "rotate",
          "parameters": [
            [20, 0, 20]
          ]
        },
      ],
    }
  ]

Fetching with alternative URI schemes
-------------------------------------

The framework uses URI schemes to define the path of the sources that are loaded.
By default it tries to load from the project local folder, using the ``file`` URI scheme (``"file://"``).
It is possible to fetch morphologies directly from `neuromorpho.org
<https://neuromorpho.org>`_ using the NeuroMorpho scheme (``"nm://"``). Then, associate it to the :guilabel:`morphologies` list of
your ``top_type``:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 11-21,40-60
    :emphasize-lines: 9-10

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 22-32

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
    :lines: 39-44

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
