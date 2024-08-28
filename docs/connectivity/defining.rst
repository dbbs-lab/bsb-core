====================
Connectivity
====================

The connection between cells is managed with ``ConnectivityStrategy`` objects. A strategy
present all the rules to apply for the definition of connection among cells. In BSB is implemented
a set of strategies ready to use (here is a :doc:`list </connectivity/connection-strategies>`), but it also possible to build a customised strategy.
Once created the connectivity are stored in :class:`ConnectivitySets <.storage.interfaces.ConnectivitySet>`.

Adding a connection type
========================

Connections are defined in the configuration under the ``connectivity`` block:

.. tab-set-code::
    .. code-block:: json

       "connectivity": {
         "type_A_to_type_B": {
           "strategy": "bsb.connectivity.VoxelIntersection",
           "presynaptic": {
             "cell_types": ["type_A"]
           },
           "postsynaptic": {
             "cell_types": ["type_B"]
           }
         }
       }
    .. code-block:: python

      config.connectivity.add(
        "type_A_to_type_B",
        strategy="bsb.connectivity.VoxelIntersection",
        presynaptic=dict(cell_types=["type_A"]),
        postsynaptic=dict(cell_types=["type_B"]),
      )


* :guilabel:`strategy`: Which :class:`~.connectivity.strategy.ConnectionStrategy` to load.
* :guilabel:`pre`/:guilabel:`post`: The pre/post-synaptic
  :class:`hemitypes <.connectivity.strategy.Hemitype>`:

  * :guilabel:`cell_types`: A list of cell types.
  * :guilabel:`labels`: (optional) a list of labels to filter the cells by
  * :guilabel:`morphology_labels`: (optional) a list of labels that filter which pieces
    of the morphology to consider when forming connections (such as ``axon``,
    ``dendrites``, or any other label you've created)

What each connection type does depends entirely on the selec

The framework will load the specified :guilabel:`strategy`, and will ask the strategy
to determine the regions of interest, and will queue up one parallel job per region of
interest. In each parallel job, the data generated during the placement step is used to
determine presynaptic to postsynaptic connection locations.

Targetting subpopulations using cell labels
===========================================

Each hemitype (:guilabel:`presynaptic` and :guilabel:`postsynaptic`) accepts an
additional list of labels to filter the cell populations by. This can be used to
connect subpopulations of cells that are labelled with any of the given labels:

.. tab-set-code::

    .. code-block:: json

      {
        "connectivity": {
          "type_A_to_type_B": {
            "strategy": "bsb.connectivity.AllToAll",
            "presynaptic": {
            "cell_types": ["type_A"],
            "labels": ["subgroup1", "subgroup2"]
          },
          "postsynaptic": {
            "cell_types": ["type_B"]
          }
        }
      }

    .. code-block:: python

      config.connectivity.add(
        "type_A_to_type_B",
        strategy="bsb.connectivity.AllToAll",
        presynaptic=dict(cell_types=["type_A"],labels=["subgroup1","subgroup2"]),
        postsynaptic=dict(cell_types=["type_B"]),
      )

This snippet would connect only the cells of ``type_A`` that are labelled with either
``subgroup1`` or ``subgroup2``, to all of the cells of ``type_B``.

Specifying subcellular regions using morphology labels
======================================================

You can also specify which regions on a morphology you're interested in connecting. By
default axodendritic contacts are enabled, but by specifying different :guilabel:`morphology_labels`
you can alter this behavior. This example lets you form dendrodendritic contacts:

.. tab-set-code::
    .. code-block:: json

      {
        "connectivity": {
         "type_A_to_type_B": {
            "strategy": "bsb.connectivity.VoxelIntersection",
            "presynaptic": {
              "cell_types": ["type_A"],
              "morphology_labels": ["dendrites"]
            },
            "postsynaptic": {
              "cell_types": ["type_B"],
             "morphology_labels": ["dendrites"]
            }
          }
        }
      }

    .. code-block:: python

      config.connectivity.add(
        "type_A_to_type_B",
        strategy="bsb.connectivity.VoxelIntersection",
        presynaptic=dict(cell_types=["type_A"],morphology_labels= ["dendrites"]),
        postsynaptic=dict(cell_types=["type_B"],morphology_labels= ["dendrites"]),
      )


In general this works with any label that is present on the morphology. You could
process your morphologies to add as many labels as you want, and then create different
connectivity targets.

Define an order for the execution of your Connectivity Strategies
=================================================================

It may be necessary to establish certain connections only after specific strategies have been executed.
In such cases, you can define a list of strategies as dependencies.
For example, you can create a :guilabel:`secondary_connection` that is established only after the
:guilabel:`type_A_to_type_B` connectivity has been completed.

.. tab-set-code::
    .. code-block:: json

      {
        "connectivity": {
         "secondary_connection": {
            "strategy": "bsb.connectivity.VoxelIntersection",
            "presynaptic": {
              "cell_types": ["type_B"],
            },
            "postsynaptic": {
              "cell_types": ["type_C"],
            },
            "depends_on": ["type_A_to_type_B"]
          }
        }
      }

    .. code-block:: python

      config.connectivity.add(
        "secondary_connection",
        strategy="bsb.connectivity.VoxelIntersection",
        presynaptic=dict(cell_types=["type_B"]),
        postsynaptic=dict(cell_types=["type_C"]),
        depends_on= ["type_A_to_type_B"]
      )

