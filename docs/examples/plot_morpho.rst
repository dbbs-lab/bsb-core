Display morphologies from a compiled network
============================================

There are several tools that you can use to visualize morphologies. Among them, you can use:

- `NeuroTessMesh <https://github.com/vg-lab/NeuroTessMesh/tree/master>`_
- `Brayns <https://github.com/BlueBrain/Brayns>`_ (efficient for large scale networks)
- `NeuroMorphoVis <https://github.com/BlueBrain/NeuroMorphoVis>`_
- Matplotlib but only for a couple of morphologies

Here is a small code snippet showing how you can extract and display points of the morphology in 3D:

.. literalinclude:: /../examples/plotting/plotting_with_branch_colors.py
    :language: python
    :lines: 2-

Remember that you can also load morphologies directly from their swc files:

.. code-block:: python

    from bsb import parse_morphology_file
    morpho = parse_morphology_file("path/to/file.swc")
