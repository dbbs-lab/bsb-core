
##################
Your first network
##################

Make sure you completed the :doc:`Installation guide</getting-started/installation>` before
running the example in this section.

.. note::

    | This guide aims to get your first model running with the minimum number of steps.
    | If you would like to familiarize yourself with the core concepts and get a more top level
      understanding first, check out the :doc:`/config/files` before you continue.

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

* ``network_configuration.json``: your configuration file. The components are declared and
  parametrized here.
* A ``pyproject.toml`` file: your project settings are declared here.
* A ``placement.py`` and ``connectome.py`` file to put your code in.

The configuration contains already a :guilabel:`partition` ``base_layer``, a :guilabel:`cell_type`
``base_type`` and a :guilabel:`placement` strategy ``example_placement``.
These minimal components are enough to *compile* your first network. You can do this from the terminal
or with Python:

.. tab-set-code::

  .. code-block:: bash

    bsb compile --verbosity 3

  .. code-block:: python

    import bsb.options
    from bsb import Scaffold, parse_from_file

    bsb.options.verbosity = 3
    config = parse_from_file("network_configuration.json", parser="json")
    scaffold = Scaffold(config)
    scaffold.compile()

Here, the ``verbosity`` flag increases the amount of output (logs) that is generated when BSB is running,
to follow along or troubleshoot.

When BSB compiles a `Scaffold`, it extracts and runs the reconstruction pipeline defined in the `Configuration` and
stores each step's results into the `Storage` (as explained in the :ref:`previous section <get-started>`).

The compile command (or python script) should produce a file ``"network.hdf5"`` located in your project
folder if BSB could parse the configuration file and complete the reconstruction. This file should
contain your network (configuration and storage) after reconstruction.

.. note::

    The configuration file can be written in either ``json`` or ``yaml`` format;
    By default, the :guilabel:`new` command uses the yaml format unless the ``--json``
    flag is set.

If you prefer, instead of loading the configuration from a file, you can create your configuration
directly in Python code with a ``Configuration`` object:

  .. code-block:: python

    import bsb.options
    from bsb import Scaffold, Configuration

    bsb.options.verbosity = 3
    config = Configuration.default(storage=dict(engine="hdf5", root="network.hdf5"))
    # Implement your code here

    scaffold = Scaffold(config)
    scaffold.compile()

.. _getting-started-configurables:

Define starter components
=========================

Network
-------

The ``network`` component describes the global spatial properties of your circuit,
including its size along the three dimensions :guilabel:`x`, :guilabel:`y`, :guilabel:`z`
(in µm).

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json
    :lines: 7-11

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python
    :lines: 7-9

Topology
--------

Your network model needs a description of its shape, which is called the topology of the
network. The topology consists of 2 components: :doc:`Regions </topology/regions>`
and :doc:`Partitions </topology/partitions>`.
Regions combine multiple partitions and/or regions together, in a hierarchy, all the way
up to a single topmost region, while partitions are exact pieces of volume that can be
filled with cells.

To get started, we will add a second layer ``top_layer``, and a region ``brain_region``:

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json
    :lines: 12-27

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python
    :lines: 11-20

The :guilabel:`type` of the ``brain_region`` is ``stack``. This means it will place its
children stacked on top of each other. The :guilabel:`type` of ``base_layer`` is
``layer``. Layers specify their size in one dimension, and fill up the space in the other
dimensions. See the :doc:`topology section</topology/intro>` for more explanation on
these components.

.. warning::
    BSB checks the configuration for errors each time the latter is modified. Now, in the
    Python code implementation, we are adding components one by one. This means that if
    one component refers to another, this latter should already in the configuration.
    That is why, in the python code implementation, we created the partitions before the
    region because the region uses references to the partitions' name.

Cell types
----------

The :doc:`Cell Types </cells/intro>` defines of populations of cells.
In the simplest case, you can define a ``cell type`` by its soma :guilabel:`radius` and
the number of cells to place using either a :guilabel:`density` value, or a fixed
:guilabel:`count`, or another
:doc:`placement indication </placement/placement-indicators>`.

To populate our new ``top_layer``, we will create an extra cell type ``top_type``; this
time we want to a place 40 of these cells and their soma :guilabel:`radius` of ``7``.

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json
    :lines: 28-41

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python
    :lines: 22-29


Placement
---------

The :doc:`placement </placement/intro>` blocks is in charge of placing cells in the
partitions using the cell type indications. For each placement component, you should
specify the placement :guilabel:`strategy` to use, the list of :guilabel:`cell_types`
names to place and the list of :guilabel:`partitions` in which you want the placement
to happen.

Now that we have defined our new ``top_type``, we should place it in our ``top_layer``:

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json
    :lines: 42-53

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python
    :lines: 31-42

We added here the ``top_placement`` that place cells soma randomly within their respective partition.

You should now try to compile your network to check if you did no mistake:

.. tab-set-code::

  .. code-block:: bash

    bsb compile -v 3  --clear

  .. code-block:: python

    # bsb.options.verbosity = 3  # if not set previously
    scaffold.compile(clear=True)

.. note::

 We are using the short forms ``-v`` of the CLI options ``verbosity``.
 You can use ``bsb --help`` to inspect the :doc:`CLI options </cli/options>`.

.. warning::

  We pass the ``clear`` flag to indicate that existing data may be overwritten. See
  :ref:`storage_control` for more flags to deal with existing data.

Each placement strategy generates a `PlacementSet` in the `Storage` that you can access from the `Scaffold` object
(see :doc:`this section </placement/placement-set>` for more info).


Connectivity
------------

The :doc:`connectivity </connectivity/defining>` component contains the blocks that specify
connections between systems of cell types.
For each :guilabel:`connectivity` component, you should specify the connection :guilabel:`strategy` and
for both :guilabel:`presynaptic` (source) and :guilabel:`postsynaptic` (target) groups, provide the
list of :guilabel:`cell_types` names to connect.

Here, we are going to connect all ``base_type`` cells to all ``top_type`` cells.

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json
    :lines: 54-65

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python
    :lines: 44-49

Recompile the network once more, now it will also contain your connections! With your
cells and connections in place, you are ready to move to the next stage.

.. note::
  For Python, the `compile` function should be called (only once) at the end of your script,
  once the configuration is complete.

Each connection strategy generates a `ConnectivitySet` in the `Storage` for each pair of neurons
that you can access from the `Scaffold` object (see :doc:`this section </connectivity/connectivity-set>` for more info).
Here, the name of the `ConnectivitySet` corresponds to the connection component (``A_to_B``) because
there is only one pair of :guilabel:`cell_type`.

.. warning::
  If you have more than one pair of cell types connected through the same connection strategy, then the name of
  the `ConnectivitySet` is ``NameOfTheComponent`` _ ``NameOfPreType`` _ ``NameOfPostType`` (learn more `here`).

Final configuration file
------------------------

.. tab-set-code::

  .. literalinclude:: configs/getting-started.json
    :language: json

  .. literalinclude:: /../examples/tutorials/getting_started.py
    :language: python

What is next?
=============
Learn how to extract the data from your produced `Scaffold` through :doc:`this tutorial <basics>`.
