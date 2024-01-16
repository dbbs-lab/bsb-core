Morphology parsers
==================

``BsbParser``
-------------

This is the default parser. It only parses SWC files, but it allows you to create as many
different SWC tags as you want, and can assign labels based on these tags. This allows you
to create a rich annotation system on top of the SWC format to denote dendritic spines,
axonal myelination with saltatory conduction, or any other morphological feature.

.. autoconfig:: bsb.morphologies.parsers.parser.BsbParser

The default SWC tags are ``1`` for ``soma``, ``2`` for ``axon``, and ``3`` for ``dendrites``.
You can add/overwrite tags by setting the :guilabel:`tags` attribute:

.. code-block:: json

  {
    "parser": "bsb",
    "tags": {
      4: ["dendrites", "apical_dendrites"],
      5: ["dendrites", "basal_dendrites"],
      6: ["axon", "axon_initial_segment"],
      7: ["axon", "axon_hillock"],
      8: ["axon", "myelin"],
      9: ["axon", "myelin", "node_of_ranvier"]
    }
  }

The soma is usually approximated as a stack of cylinders, this means that 


``MorphIOParser``
-----------------

.. autoconfig:: bsb.morphologies.parsers.parser.MorphIOParser