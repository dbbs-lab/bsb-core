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
It is possible to load local files or to fetch from remote sources, like NeuroMorpho.

Using local files
-----------------
You can declare source morphologies in the root :guilabel:`morphologies` list:

.. tab-set-code::

  .. code-block:: json

       "morphologies": [
         "neuron_A.swc"
       ],

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 18

.. hint::

    To follow along, you can download a morphology from NeuroMorpho and save it as ``neuron_A.swc`` locally.

In this case a morphology is created from ``neuron_A.swc`` with the name ``"neuron_A"``.
As a second step, we associate this morphology to the ``top_type`` by referencing it by name
in :guilabel:`cell_types.top_type.spatial.morphologies`:

.. tab-set-code::

  .. literalinclude:: include_morphos.json
    :language: json
    :lines: 41-50
    :emphasize-lines: 6-9

  .. literalinclude:: include_morphos.py
    :language: python
    :lines: 21


By default the name assigned to the morphology is the file name without ``.swc`` extension. To
change the name we can use the extended notation, specifying a node with a :guilabel:`name` and :guilabel:`file`:

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
morphology. Pipelines can be added by adding a :guilabel`pipeline` list to the morphology node.
Each item in the list may either be a string reference to an importable function or a method of
the :class:`~bsb.morphologies.Morphology` class. To pass parameters, use a node with the
function reference placed in the guilabel:`func` attribute, and a :guilabel:`parameters` list:

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

.. info::

  Parameters are passed positionally, keyword arguments must be passed in the order they appear
  in in the signature. If your target function has a complicated signature or keyword-only
  arguments, create a wrapping function and target that instead.

Fetching with alternative URI schemes
-------------------------------------

The framework uses URI schemes to define the path of the sources that are loaded.
By default it tries to load from the project local folder, using the ``file`` URI scheme (``"file://"``).
It is possible to fetch morphologies directly from `neuromorpho.org
<https://neuromorpho.org>`_ using the NeuroMorpho scheme (``"nm://"``):

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
