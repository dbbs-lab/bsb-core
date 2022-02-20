###############
Getting Started
###############

Follow the :doc:`/usage/installation`:

* Set up a new environment
* Install the software into the environment

.. note::

	This guide aims to get your first model running with the bare minimum steps. If you'd
	like to familiarize yourself with the core concepts and get a more top level
	understanding first, check out the :doc:`./top-level-guide` before you continue.

There are 2 ways of building models using the Brain Scaffold Builder (BSB), the first is
through **configuration**, the second is **scripting**. The 2 methods complement each
other so that you can load the general model from a configuration file and then layer on
more complex steps under your full control in a Python script. Be sure to take a quick
look at each code tab to see the equivalent forms of configuration coding!

Create a project
================

Use the command below to create a new project directory and some starter files:

.. code-block:: bash

  bsb new my_first_model

You'll be asked some questions; enter a value, or just press Enter to select the default
value. The proposed ``starting_example.json`` template is a good first configuration file.

The project now contains a couple of important files:

* A configuration file, your components are declared and parametrized here.
* A ``pyproject.toml`` file, your project settings are declared here.
* A ``placement.py`` and ``connectome.py`` file, to put your code in.

Take a look at ``starting_example.json``; it contains a nondescript ``brain_region``, a
``base_layer``, a ``base_type`` and an ``example_placement``. These minimal components are
enough to *compile* your first network. You can do this from the CLI or Python:

.. tab-set-code::

  .. code-block:: bash

    bsb compile --verbosity 3 --plot

  .. code-block:: python

    from bsb.core import Scaffold
    from bsb.config import from_json
    from bsb.plotting import plot_network
    import bsb.options

    bsb.options.verbosity = 3
    config = from_json("starting_example.json")
    scaffold = Scaffold(config)
    scaffold.compile()
    plot_network(scaffold)

The ``verbosity`` helps you follow along what instructions the framework is executing and
``plot`` should.. open a plot |:slight_smile:|.

.. rubric:: What next?

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item-card:: :octicon:`flame;1em;sd-text-warning` Continue getting started
	    :link: getting-started-configurables
	    :link-type: ref

	    Follow the rest of the guide for basics on as ``CellTypes``, ``Placement`` blocks,
	    ``Connectivity`` blocks and ``Simulations``.

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Components
	    :link: components
	    :link-type: ref

	    Learn how to write your own components to e.g. place or connect cells.

    .. grid-item-card:: :octicon:`database;1em;sd-text-warning` Simulations
	    :link: simulations
	    :link-type: ref

	    Learn how to simulate your network models

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
	    :link: examples
	    :link-type: ref

	    View examples explained step by step

    .. grid-item-card:: :octicon:`package-dependents;1em;sd-text-warning` Plugins
	    :link: plugins
	    :link-type: ref

	    Learn to package your code for others to use!

    .. grid-item-card:: :octicon:`octoface;1em;sd-text-warning` Contributing
	    :link: https://github.com/dbbs-lab/bsb

	    Help out the project by contributing code.

.. _getting-started-configurables:

Define starter components
=========================

Topology
--------

Your network model needs a description of its shape, which is called the topology of the
network. The topology exists of 2 types of components: :class:`Regions
<.topology.region.Region>` and :class:`Partitions <.topology.partition.Partition>`.
Regions combine multiple partitions and/or regions together, in a hierarchy, all the way
up to a single topmost region, while partitions are exact pieces of volume that can be
filled with cells.

To get started, we'll add a ``cortex`` region, and populate it with a ``base_layer``:

.. code-block:: json

  {
    "regions": {
      "cortex": {
        "origin": [0.0, 0.0, 0.0],
				"partitions": ["base_layer"]
      }
    },
    "partitions": {
      "base_layer": {
				"type": "layer",
        "thickness": 100
      }
    }
  }

The ``cortex`` does not specify a region :guilabel:`type`, so it is a group. The
:guilabel:`type` of ``base_layer`` is ``layer``, they specify their size in 1 dimension,
and fill up the space in the other dimensions. See :doc:`/topology/intro` for more
explanation on topology components.

Cell types
----------

The :class:`~.objects.cell_type.CellType` is a definition of a cell population. During
placement 3D positions, optionally rotations and morphologies or other properties will be
created for them. In the simplest case you define a soma :guilabel:`radius` and
:guilabel:`density` or fixed :guilabel:`count`:

.. code-block:: json

  {
    "cell_types": {
      "cell_type_A": {
        "spatial": {
          "radius": 7,
					"density": 1e-3
        }
      },
      "cell_type_B": {
        "spatial": {
          "radius": 7,
					"count": 10
        }
      }
    }
  }

Placement
---------

.. code-block:: json

	{
		"placement": {
			"cls": "bsb.placement.ParticlePlacement",
			"cell_types": ["cell_type_A", "cell_type_B"],
			"partitions": ["base_layer"]
		}
	}

The ``placement`` blocks use the cell type indications to place cell types into
partitions. You can use :class:`PlacementStrategies
<.placement.strategy.PlacementStrategy>` provided out of the box by the BSB or your own
component by setting the :guilabel:`cls`. The
:class:`~bsb.placement.particle.ParticlePlacement` considers the cells as somas and
bumps them around as repelling particles until there is no overlap between the somas.

At this point you can take another look at your network:

.. code-block:: bash

	bsb compile -v 3 -p

.. note::

	We're using the short forms ``-v`` and ``-p`` of the CLI options ``--verbosity`` and
	``--plot``, respectively. You can use ``bsb --help`` to inspect the CLI options.

Connectivity
------------

.. code-block:: json

  {
    "connectivity": {
      "A_to_B": {
        "cls": "bsb.connectivity.AllToAll",
        "from_type": "cell_type_A",
        "to_type": "cell_type_B"
      }
    }
  }

<More conn info>

At this point compiling the network generates cell positions and connections and
we can move to the :ref:`simulations` stage.
