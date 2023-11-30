
!!!!!!!!!  CURRENTLY IN EXCLUDEPATTERNS  !!!!!!!!!
!!   This file is ignored while we restructure  !!
!!             the documentation                !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

######
Labels
######

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
        "class": "my_model.postprocessing.LabelCellA",
        "targets": ["cell_A"]
      }
    }
  }

For more information on linking your Python classes to the configuration file see
:doc:`/config/intro`.

Example of a Python class for labeling neurons.
-----------------------------------------------

.. code-block:: python

  from bsb.postprocessing import PostProcessingHook


  class LabelCellA(PostProcessingHook):
      ''' Subdivide the cell_A population into 2 subpopulations '''

      def after_placement(self):
          ids = self.scaffold.get_cells_by_type("cell_A")[:, 0]
          number_of_cells = len(ids)
          subpopulation_1 = ids[0:int(number_of_cells/2)]
          subpopulation_2 = ids[int(number_of_cells/2):]

          self.scaffold.label_cells(
              subpopulation_1, label="cell_A_type_1",
          )
          self.scaffold.label_cells(
              subpopulation_2, label="cell_A_type_2",
          )

In this example, we can see that the ``LabelCellA`` class must inherit from
``PostProcessingHook`` and it must specify a method ``after_placement`` in which the
neural population ``cell_A`` is subdivided into two populations:

* ``subpopulation_1`` contains the ids of the first half of the population
* ``subpopulation_2`` contains the ids of the second half of the population

Then, these ids are used to assign the labels ``cell_A_type_1`` and ``cell_A_type_2`` to
``subpopulation_1`` and ``subpopulation_2``, respectively.
