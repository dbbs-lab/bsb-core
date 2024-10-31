How to label neurons
********************

After placing cells inside the scaffold model, it is possible to define postprocessing
functions that modify some features of the scaffold. For instance, it is possible to
define a function that, given a specific cell type, assigns a label to each cell belonging
to that cell type  (e.g., subdivide a certain population into different subpopulations
according to their position in the 3D space.)

Postprocessing functions can be configured in the ``after_placement`` dictionary of the
root node of the configuration file, specifying each postprocessing function with its
name, e.g. "Labels":

.. code-block:: json

   {
     "after_placement": {
       "Labels": {
         "strategy": "my_model.postprocessing.LabelCellA",
         "cell_type": "cell_A"
       }
     }
   }

For more information on linking your Python classes to the configuration file see
:doc:`this section </config/nodes>`.

Example of a Python class for labeling neurons
----------------------------------------------

.. literalinclude:: /../examples/cells/label_cells.py
  :language: python
  :lines: 2-

In this example, we can see that the ``LabelCellA`` class must inherit from
``AfterPlacementHook`` and it must specify a method ``postprocess`` in which the
neural population ``cell_A`` is subdivided into two populations.

Here, along the chosen axis, cells placed above the mean position of the population
will be assigned the label ``cell_A_type_1`` and the rest ``cell_A_type_2``.

You can then filter back these cells like so:

.. code-block:: python

    from bsb import from_storage

    scaffold = from_storage("my_network.hdf5")
    ps = scaffold.get_placement_set("cell_A")
    subpopulation_1 = ps.get_labelled(["cell_A_type_1"])
    subpopulation_2 = ps.get_labelled(["cell_A_type_2"])

    # or alternatively directly filter when loading the placement set
    ps_1 = scaffold.get_placement_set("cell_A", labels=["cell_A_type_1"])
    ps_2 = scaffold.get_placement_set("cell_A", labels=["cell_A_type_2"])
