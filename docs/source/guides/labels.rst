######
Labels
######

How to label neurons
********************

After placing cells inside the scaffold model, it is possible to define postprocessing functions
that modify some features of the scaffold. For instance, it is possible to define a function
that, taken a specific cell type, assign a label to each cell belonging to that cell type 
(e.g., subdivide a certain population into different subclasses according to their position in the 3D space.)

Postprocessing functions can be configured in the ``after_placement`` dictionary of the root node of the
configuration file, specifying each postprocessing function with its name, e.g. "Labels":

.. code-block:: json

  {
    "after_placement": {
      "Labels": {
        "class": "my_model.postprocessing.LabelCellA",
        "targets": ["cell_A"]
      }
    }
  }

For more information on linking your Python classes to the configuration file see :doc:`/config/intro`.

Example of a Python class for labeling neurons.
-----------------------------------------------

.. code-block:: python

  from bsb.helpers import ConfigurableClass

  class PostProcessingHook(ConfigurableClass):
      def validate(self):
          pass

      def after_placement(self):
          raise NotImplementedError(
              "`after_placement` hook not defined on " + self.__class__.__name__
          )

      def after_connectivity(self):
          raise NotImplementedError(
              "`after_connectivity` hook not defined on " + self.__class__.__name__
          )

  class LabelCellA(PostProcessingHook):
      ''' Subdivide the cell_A population into 2 subclasses '''

      def after_placement(self):
          ids = self.scaffold.get_cells_by_type("cell_A")[:, 0]
          number_of_cells = len(ids)
          subclass_1 = ids[0:int(number_of_cells/2)]
          subclass_2 = ids[int(number_of_cells/2):]

          self.scaffold.label_cells(
              subclass_1, label="cell_A_type_1",
          )
          self.scaffold.label_cells(
              subclass_2, label="cell_A_type_2",
          )

In this example, we can see that the ``LabelCellA`` class must inherit from ``PostProcessingHook``
and it must specify a method ``after_placement`` in which the neural population ``cell_A`` is subdivided
into two populations:

* ``subclass_1`` contains the ids of the first half of the population
* ``subclass_2`` contains the ids of the second half of the population
Then, these ids are used to assign the labels ``cell_A_type_1`` and ``cell_A_type_2`` to ``subclass_1`` and
``subclass_2``, respectively.