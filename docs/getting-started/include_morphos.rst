.. _include_morphos:

Adding morphologies
===================

.. note::

    This guide is a continuation of the |:books:| :doc:`Getting Started guide </getting-started/getting-started>`.

.. hint::

    To follow along, download 2 morphologies from
    `NeuroMorpho <https://neuromorpho.org/>`_ and save them as ``neuron_A.swc`` and
    ``neuron2.swc`` locally.

Previously we constructed a stacked double layer topology, with 2 cell types. We then
connected them in an all-to-all fashion. The next step assigns
:doc:`morphologies </morphologies/intro>` to our cells, and connects the cells based on
the intersection of their morphologies!

Morphologies can be loaded from local files or to fetch from remote sources, like NeuroMorpho.

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

In this case a morphology is created from ``neuron_A.swc`` and given the name ``"neuron_A"``.
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


By default the name assigned to the morphology is the file name without its extension (here ``.swc``). To
change the name we can use a node with a :guilabel:`name` and :guilabel:`file`:

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

.. note::

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


Morphology intersection
-----------------------

Now that we have assigned morphologies to our cell types, we can use morphology-based
connection strategies such as :class:`~.connectivity.detailed.voxel_intersection.VoxelIntersection`:

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
