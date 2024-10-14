
##################
Your first network
##################

Follow the :doc:`/getting-started/installation`:

* Set up a new environment
* Install the software into the environment

.. note::

	This guide aims to get your first model running with the bare minimum steps. If you'd
	like to familiarize yourself with the core concepts and get a more top level
	understanding first, check out the :doc:`/components/intro` before you continue.

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

    bsb compile --verbosity 3

  .. code-block:: python

    import bsb.options
    from bsb import Scaffold, parse_from_file

    bsb.options.verbosity = 3
    config = parse_from_file("network_configuration.json",parser="json")
    network = Scaffold(config)
    network.compile()

The ``verbosity`` flag increases the amount of output that is generated, to follow along
or troubleshoot.

.. note::

    The configuration file can be written in either ``json`` or ``yaml`` format;
    simply specify the format in the :guilabel:`new` command options:


Alternatively, if you prefer working with Python code, you can start by creating a ``Configuration`` object:

  .. code-block:: python

    import bsb.options
    from bsb import Scaffold, Configuration

    config = Configuration.default(storage={"engine": "hdf5"})




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
    :lines: 7-16

The :guilabel:`type` of the ``brain_region`` is ``stack``. This means it will place its
children stacked on top of each other. The :guilabel:`type` of ``base_layer`` is
``layer``. Layers specify their size in 1 dimension, and fill up the space in the other
dimensions. See :doc:`/topology/intro` for more explanation on topology components.

Cell types
----------

The :doc:`Cell Type </cells/intro>` is a definition of a cell population. During
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
    :lines: 20-25


The :doc:`placement </placement/intro>` blocks is in charge of placing cells in the partitions using the cell type indications.
You can specify the strategy to use  by setting the :guilabel:`strategy` attribute.
Here we use  the strategy :guilabel:`ParticlePlacement` that considers the cells as spheres and
bumps them around as repelling particles until there is no overlap between them.

Take another look at your network:

.. code-block:: bash

  bsb compile -v 3  --clear

.. note::

 We're using the short forms ``-v`` of the CLI options ``--verbosity``.
 You can use ``bsb --help`` to inspect the CLI options.

.. warning::

  We pass the ``--clear`` flag to indicate that existing data may be overwritten. See
  :ref:`storage_control` for more flags to deal with existing data.


Connectivity
------------

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json
    :lines: 54-64

  .. literalinclude:: getting_started.py
    :language: python
    :lines: 26-31


The :doc:`connectivity </connectivity/defining>` blocks specify connections between systems of cell types. They can
create connections between single or multiple cell types for both pre and post synaptic groups.

Regenerate the network once more, now it will also contain your connections! With your
cells and connections in place, you're ready to move to the next stage.




Recap
-----

.. tab-set-code::

  .. literalinclude:: getting-started.json
    :language: json

  .. literalinclude:: getting_started.py
    :language: python
