.. _get-started:

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

The framework supports both declarative statements in configuration formats, or Python
code. Be sure to take a quick look at each code tab to get a feel for the equivalent forms
of configuration coding!

Create a project
================

Use the command below to create a new project directory and some starter files:

.. code-block:: bash

  bsb new my_first_model --quickstart --json
  cd my_first_model

The project now contains a couple of important files:

* ``network_configuration.json``: your components are declared and parametrized here.
* A ``pyproject.toml`` file: your project settings are declared here.
* A ``placement.py`` and ``connectome.py`` file to put your code in.

The configuration contains a ``base_layer``, a ``base_type`` and an ``example_placement``.
These minimal components are enough to *compile* your first network. You can do this from
the CLI or Python:

.. tab-set-code::

  .. code-block:: bash

    bsb compile --verbosity 3 --plot

  .. literalinclude:: getting_started.py
    :language: python
    :lines: -7,32-

The ``verbosity`` flag increases the amount of output that is generated, to follow along
or troubleshoot. The ``plot`` flags opens a plot |:slight_smile:|.

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

To get started, we'll add a second layer ``top_layer``, and a region ``brain_region``
which will stack our layers on top of each other:

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json
    :lines: 12-29

  .. literalinclude:: getting_started.py
    :language: python
    :lines: 9-17

The :guilabel:`type` of the ``brain_region`` is ``stack``. This means it will place its
children stacked on top of each other. The :guilabel:`type` of ``base_layer`` is
``layer``. Layers specify their size in 1 dimension, and fill up the space in the other
dimensions. See :doc:`/topology/intro` for more explanation on topology components.

Cell types
----------

The :class:`~.cell_types.CellType` is a definition of a cell population. During
placement 3D positions, optionally rotations and morphologies or other properties will be
created for them. In the simplest case you define a soma :guilabel:`radius` and
:guilabel:`density` or fixed :guilabel:`count`:

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json
    :lines: 30-43

  .. literalinclude:: getting_started.py
    :language: python
    :lines: 18


Placement
---------

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json
    :lines: 44-55

  .. literalinclude:: getting_started.py
    :language: python
    :lines: 19-24


The ``placement`` blocks use the cell type indications to place cell types into
partitions. You can use other :class:`PlacementStrategies
<.placement.strategy.PlacementStrategy>` by setting the :guilabel:`strategy` attribute.
The BSB offers some strategies out of the box, or you can implement your own. The
:class:`~bsb.placement.particle.ParticlePlacement` considers the cells as spheres and
bumps them around as repelling particles until there is no overlap between them. The data
is stored in :class:`PlacementSets <.storage.interfaces.PlacementSet>` per cell type.

Take another look at your network:

.. code-block:: bash

  bsb compile -v 3 -p --clear

.. note::

	We're using the short forms ``-v`` and ``-p`` of the CLI options ``--verbosity`` and
	``--plot``, respectively. You can use ``bsb --help`` to inspect the CLI options.

.. warning::

  We pass the ``--clear`` flag to indicate that existing data may be overwritten. See
  :ref:`storage_control` for more flags to deal with existing data.


Connectivity
------------

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json
    :lines: 56-66

  .. literalinclude:: getting_started.py
    :language: python
    :lines: 25-30


The ``connectivity`` blocks specify connections between systems of cell types. They can
create connections between single or multiple pre and postsynaptic cell types, and can
produce one or many :class:`ConnectivitySets <.storage.interfaces.ConnectivitySet>`.

Regenerate the network once more, now it will also contain your connections! With your
cells and connections in place, you're ready to move to the :ref:`simulations` stage.

.. rubric:: What next?

.. grid:: 1 1 2 2
    :gutter: 1

    .. grid-item-card:: :octicon:`flame;1em;sd-text-warning` Continue getting started
	    :link: include_morphos
	    :link-type: ref

	    Follow the rest of the guides and learn how to include morphologies.

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

    .. grid-item-card:: :octicon:`mark-github;1em;sd-text-warning` Contributing
	    :link: https://github.com/dbbs-lab/bsb

	    Help out the project by contributing code.

Recap
-----

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json

  .. literalinclude:: getting_started.py
    :language: python
