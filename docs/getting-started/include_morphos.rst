.. _include_morphos:

###################
Adding morphologies
###################

.. note::

    This guide is a continuation of the
    :doc:`Getting Started guide </getting-started/getting-started_reconstruction>`.

Previously, we constructed a stacked double layer topology, with 2 cell types. We then
connected these cell type populations in an all-to-all fashion.

In this tutorial, we are going to assign :doc:`morphologies </morphologies/intro>` to our
cells, and connects the cells based on the intersection of their morphologies!
You will learn how to load morphologies from local files or to fetch
from remote sources, like NeuroMorpho, using the BSB.

| But first, we need actual morphology files.
| Download your 2 favorite morphologies from `NeuroMorpho <https://neuromorpho.org/>`_
  in the swc file format and save them as ``neuron_A.swc`` and ``neuron2.swc`` in your
  project folder.


Using local files
=================

:guilabel:`morphologies` is a list component of the BSB configuration responsible
to fetch and load morphology files. Here is the minimal configuration example to add a
morphology to the scaffold:

.. tab-set-code::

  .. literalinclude:: configs/include_morphos.yaml
    :language: yaml
    :lines: 9-10

  .. code-block:: json

       "morphologies": ["neuron_A.swc"]

  .. code-block:: python

    config.morphologies = ["neuron_A.swc"]

In this case, a morphology is created from ``neuron_A.swc`` and given the name ``"neuron_A"``.
By default the name assigned to the morphology is the file name without its extension (here ``.swc``).

Next, we need to associate this morphology to one cell type, here the ``base_type``, by
referencing it by name in :guilabel:`cell_types.base_type.spatial.morphologies`:

.. tab-set-code::

  .. literalinclude:: configs/include_morphos.yaml
    :language: yaml
    :lines: 28-34
    :emphasize-lines: 6-7

  .. literalinclude:: configs/include_morphos.json
    :language: json
    :lines: 35-42
    :emphasize-lines: 6

  .. literalinclude:: /../examples/tutorials/include_morphos.py
    :language: python
    :lines: 27-34
    :emphasize-lines: 6

.. note::

  If there are multiple morphologies per cell type, they will be assigned randomly, unless you
  specify a :ref:`MorphologyDistributor <MorphologyDistributors>`.

Let's add the second morphology but this time we will change its name with a morphology node
containing the attributes :guilabel:`name` and :guilabel:`file`:

.. tab-set-code::

  .. literalinclude:: configs/include_morphos.yaml
    :language: yaml
    :lines: 9-12
    :emphasize-lines: 3-4

  .. literalinclude:: configs/include_morphos.json
    :language: json
    :lines: 12-17
    :emphasize-lines: 3-6

  .. literalinclude:: /../examples/tutorials/include_morphos.py
    :language: python
    :lines: 22-25
    :emphasize-lines: 3

It is also possible to add a pipeline to perform transformations on the loaded
morphology. Pipelines can be added with a :guilabel:`pipeline` list component to the
morphology node.
Each item in the list may either be a string reference to a method of the
:class:`~bsb.morphologies.Morphology` class or an importable function.
If the function requires parameters, use a node with the function reference placed in the
:guilabel:`func` attribute, and a :guilabel:`parameters` list.

Here is an example what that would look like:

.. tab-set-code::

  .. code-block:: yaml

    morphologies:
      - file: my_neuron.swc
        pipeline:
          - center
          - my_module.add_axon
          - func: rotate
            rotation: [20, 0, 20]

  .. code-block:: json

    "morphologies": [
      {
        "file": "my_neuron.swc",
        "pipeline": [
          "center",
          "my_module.add_axon",
          {
            "func": "rotate",
            "rotation": [20, 0, 20]
          },
        ],
      }
    ]

  .. code-block:: python

    config.morphologies = [
      dict(
        file= "my_neuron.swc",
        pipeline=[
          "center",
          "my_module.add_axon",
          dict(func="rotate", rotation=[20, 0, 20])
        ]
      )
    ]

In this case, we created a pipeline of 3 steps:

1. Reset the origin of the morphology, using the :meth:`~.morphologies.SubTree.center` function from the
   Morphology class.
2. Run the :guilabel:`add_axon` function from the external file `my_module.py`
3. Rotate the morphology by 20 degrees along the x and z axis, using the
   :meth:`~.morphologies.SubTree.rotate` function from the Morphology class.

.. note::

  Any additional keys given in a pipeline step, such as :guilabel:`rotation` in the
  example, are passed to the function as keyword arguments.



Morphology intersection
=======================

Now that we have assigned morphologies to our cell types, we can use morphology-based
connection strategies such as :doc:`VoxelIntersection </connectivity/connection-strategies>`:

.. tab-set-code::

  .. literalinclude:: configs/include_morphos.yaml
    :language: yaml
    :lines: 55-63

  .. literalinclude:: configs/include_morphos.json
    :language: json
    :lines: 65-75

  .. literalinclude:: /../examples/tutorials/include_morphos.py
    :language: python
    :lines: 59-64

Note also that with Voxel Intersection,
you can specify which parts of the morphologies should create contacts (e.g, dendrites and axons):

.. code-block:: json

    "connectivity": {
    "A_to_B": {
      "strategy": "bsb.connectivity.VoxelIntersection",
      "presynaptic": {
        "cell_types": ["base_type"],
        "morphology_labels": ["dendrites"]
      },
      "postsynaptic": {
          "cell_types": ["top_type"],
          "morphology_labels": ["axon"]
      }
    }
  }

This happens thanks to the labels that are attached to your morphology points.

.. tip::
    Do not forget to recompile your network if you are modifying the configuration file.

Final configuration file
========================

.. tab-set-code::

  .. literalinclude:: configs/include_morphos.yaml
    :language: yaml

  .. literalinclude:: configs/include_morphos.json
    :language: json

  .. literalinclude:: /../examples/tutorials/include_morphos.py
    :language: python

What is next?
=============

Next tutorial is on :doc:`running a simulation <guide_simulation>` of your network.
