################################
Simulating networks with the BSB
################################

The BSB manages simulations by deferring as soon as possible to the simulation backends.
Each simulator has good reasons to make their design choices, fitting to their simulation
paradigm. These choices lead to divergence in how simulations are described, and each
simulator has their own niche functions. This means that if you are already familiar with
a simulator, writing simulation config should feel familiar, on top of that the BSB is
able to offer you access to each simulator's full set of features. The downside is that
you're required to write a separate simulation config block per backend.

Now, let's get started.

===================
Conceptual overview
===================

Each simulation config block needs to specify which :guilabel:`simulator` they use. Valid
values are ``arbor``, ``nest`` or ``neuron``. Also included in the top level block are the
:guilabel:`duration`, :guilabel:`resolution` and :guilabel:`temperature` attributes:

.. code-block:: json

  {
    "simulations": {
      "my_arbor_sim": {
        "simulator": "arbor",
        "duration": 2000,
        "resolution": 0.025,
        "temperature": 32,
        "cell_models": {

        },
        "connection_models": {

        },
        "devices": {

        }
      }
    }
  }

The :guilabel:`cell_models` are the simulator specific representations of the network's
:class:`cell types <.models.CellType>`, the :guilabel:`connection_models` of the network's
:class:`connectivity types <.connectivity.ConnectionStrategy>` and the :guilabel:`devices`
define the experimental setup (such as input stimuli and recorders). All of the above is
simulation backend specific and are covered in detail below.

=====
Arbor
=====

Cell models
-----------

The keys given in the :guilabel:`cell_models` should correspond to a ``cell type`` in the
network. If a certain ``cell type`` does not have a corresponding ``cell model`` then no
cells of that type will be instantiated in the network. Cell models in Arbor should refer
to importable ``arborize`` cell models. The Arborize model's ``.cable_cell`` factory will
be called to produce cell instances of the model:

.. code-block:: json

  {
    "cell_models": {
      "cell_type_A": {
        "model": "my.models.ModelA"
      },
      "afferent_to_A": {
        "relay": true
      }
    }
  }

.. note::

  *Relays* will be represented as ``spike_source_cells`` which can, through the connectome
  relay signals of other relays or devices. ``spike_source_cells`` cannot be the target of
  connections in Arbor, and the framework targets the targets of a relay instead, until
  only ``cable_cells`` are targeted.

Connection models
-----------------

todo: doc

.. code-block:: json

  {
    "connection_models": {
      "aff_to_A": {
        "weight": 0.1,
        "delay": 0.1
      }
    }
  }

Devices
-------

``spike_generator`` and ``probes``:

.. code-block:: json

  {
    "devices": {
      "input_stimulus": {
        "device": "spike_generator",
        "explicit_schedule": {
          "times": [1,2,3]
        },
        "targetting": "cell_type",
        "cell_types": ["mossy_fibers"]
      },
      "all_cell_recorder": {
        "targetting": "representatives",
        "device": "probe",
        "probe_type": "membrane_voltage",
        "where": "(uniform (all) 0 9 0)"
      }
    }
  }

todo: doc & link to targetting

====
NEST
====


======
NEURON
======
