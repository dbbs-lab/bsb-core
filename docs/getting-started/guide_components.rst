.. _guide_components:

Create your own components
==========================

.. note::
    This guide assumes that you have a good understanding of the basis of BSB; i.e., that
    you are familiar with the concept of BSB ``components``. You should have read the whole
    the getting started :doc:`section </getting-started/top-level-guide>`.

In this tutorial, we are going to guide you through the process of creating your own
components for a :doc:`Placement </placement/placement-strategies>` and a
:doc:`Connection </connectivity/connection-strategies>` strategies.
We will start from the following configuration file (corresponds to the first network file
from the getting started tutorial):

.. literalinclude:: /getting-started/configs/getting-started.json
    :language: json

Description of the strategies to implement
------------------------------------------

.. rubric:: Placement

We want here to implement a distribution placement strategy: cells will be placed within
their `Partition` following a probability ``distribution`` along a certain ``axis`` and
``direction``. For instance, let us use the
:doc:`alpha random distribution <scipy:reference/generated/scipy.stats.alpha>`.
The ``distribution`` should be a density function that produces random numbers, according
to the distance along the ``axis`` to the border of the Partition.

In short, we want to give more probability for the cells ``A`` within the ``base_layer``
to be placed closer to the border with the ``top_layer``.

.. rubric:: Connectivity

For the connectivity, we will consider that the cells ``A`` can connect to all ``B`` cells
within a sphere of ``radius`` ``100`` um. You can consider this as a simplified model of
distance based connectivity.

Components boiler plate
-----------------------

BSB components are written as Python classes. When the BSB parses your `Configuration`
component, it resolves the path to its class or function and import it. Hence, your
components should be written as
`importable modules <https://docs.python.org/3/tutorial/modules.html>`_.

In our case, we will write our `PlacementStrategy` class in the ``placement.py`` file,

.. code-block:: python

    from bsb import config, PlacementStrategy

    @config.node
    class DistributionPlacement(PlacementStrategy):

        # add your attributes here

        def place(self, chunk, indicators):
            # write your code here
            pass

Here, our class will extend from
:class:`PlacementStrategy <.placement.strategy.PlacementStrategy>` which is an
abstract class that requires you to implement the
:meth:`place <.placement.strategy.PlacementStrategy.place>` function.

Similarly, we will place our `ConnectionStrategy` class in ``connectome.py``:

.. code-block:: python

    from bsb import config, ConnectionStrategy

    @config.node
    class DistanceConnectivity(ConnectionStrategy):

        # add your attributes here

        def connect(self, presyn_collection, postsyn_collection):
            # write your code here
            pass

The :class:`ConnectionStrategy <.connectivity.strategy.ConnectionStrategy>` here
requires to implement the :meth:`connect <.connectivity.strategy.ConnectionStrategy.connect>`.

.. tip::
    Take the time to familiarize yourself with both class and functions before continuing.

.. note::
    In this example, we only implemented the required functions for both strategies but
    you can also overwrite the other functions of the interface, if you need it.
    Please refer to the documentation on the classes you want to implement for more
    information.

Note that both strategies leverage the ``@config.node``
`python decorator <https://pythonbasics.org/decorators/>`_.
The :doc:`configuration node decorator</config/nodes>` allows you to pass the parameters
defined in the configuration file to the class. It will also handle the
:doc:`type testing </config/types>` of your configuration attributes (e.g., make sure your
``radius`` parameter is a positive float number). We will see in the following sections how
to create your class configuration attributes.

At this stage, you have created 2 python class with minimal code implementation, you should
now link it to your configuration file.

Adding configuration attributes
-------------------------------

For each of our strategy, we need to pass a list of parameters to their respective class,
through our configuration file.

For our placement strategy, we need:

- a density function ``distribution``, defined by the user, from the list of
  :doc:`scipy stats <scipy:reference/stats>` functions.
- an ``axis`` along which to apply the distribution. The latter should be in [0, 1, 2]
- a ``direction`` that if set will apply the distribution along the ``"negative"`` direction,
  i.e from top to bottom (``"positive"`` direction if not set).

This translates into 3 configuration attributes that you can add to your class:

.. code-block:: python

    from bsb import config, PlacementStrategy, types

    @config.node
    class DistributionPlacement(PlacementStrategy):

        distribution = config.attr(type=types.distribution(), required=True)
        axis: int = config.attr(type=types.int(min=0, max=2), required=False, default=2)
        direction: str = config.attr(type=types.in_(["positive", "negative"]),
                                     required=False, default="positive")

        def place(self, chunk, indicators):
            # write your code here
            pass

In this case, ``distribution`` is required, and should correspond to a
:class:`distribution <.config._distributions.Distribution>` node which interface scipy distributions.
``axis`` here is an optional integer attribute with a default value set to 2.
Finally, ``direction`` is an optional string attribute that can be either the string ``"positive"``
or ``"negative"`` (see :func:`in_ <.config.types.in_>`).

Similarly, for the connectivity strategy, we need a radius positive float parameter.
This translates into the following code in our class:

.. code-block:: python

    from bsb import config, ConnectionStrategy, types

    @config.node
    class DistanceConnectivity(ConnectionStrategy):

        radius: float = config.attr(type=types.float(min=0), required=True)

        def connect(self, presyn_collection, postsyn_collection):
            # write your code here
            pass

If the parameters provided for the configuration attributes do not match the expected
types, then BSB will raise an :class:`ConfigurationError <.exceptions.ConfigurationError>`.

Finally, to import our classes in our configuration file, we will modify the
:guilabel:`placement` and :guilabel:`connectivity` blocks:

.. code-block:: json

  "placement": {
    "alpha_placement": {
      "strategy": "placement.DistributionPlacement",
      "distribution": {
        "distribution": "alpha",
        "a": 8
      },
      "axis": 0,
      "direction": "negative",
      "cell_types": ["base_type"],
      "partitions": ["base_layer"]
    },
    "top_placement": {
      "strategy": "bsb.placement.RandomPlacement",
      "cell_types": ["top_type"],
      "partitions": ["top_layer"]
    }
  },
  "connectivity": {
    "A_to_B": {
      "strategy": "connectivity.DistanceConnectivity",
      "radius": 100,
      "presynaptic": {
        "cell_types": ["base_type"]
      },
      "postsynaptic": {
          "cell_types": ["top_type"]
      }
    }
  }

Implement the python methods
----------------------------
Starting from now, we introduce the term of `Chunk` which is a volume unit used to decompose
your circuit topology into independent pieces to parallelize the circuit reconstruction
(see :doc:`this section</core/job-distribution>` for more details).

.. rubric:: Placement Strategy

The `place` function will be used here to produce and store a
:class:`PlacementSet <.storage.interfaces.PlacementSet>` for each `cell type` population
to place in the selected `Partition`.
BSB is parallelizing placement jobs for each `Chunk` concerned.

The parameters of `place` includes a dictionary linking each cell type name to its
:class:`PlacementIndicator <.placement.indicator.PlacementIndicator>`, and the `Chunk`
in which to place the cells.

We are done with the Placement! Here is how the full strategy looks like:

.. literalinclude:: /../examples/tutorials/distrib_placement.py
    :language: python

.. rubric:: Connection Strategy

Here, we are going to use the `connect` function to produce and store
:class:`ConnectivitySets <.storage.interfaces.ConnectivitySet>`.
First some definition:

| The presynaptic and postsynaptic populations to connect (each can have multiple cell type
  populations) are called :class:`Hemitype <.connectivity.strategy.Hemitype>`. An `Hemitype`
  corresponds to the interface to define a connection population and its parameters in the
  Configuration.
| The class :class:`HemitypeCollection <.connectivity.strategy.HemitypeCollection>` allows
  you to filter the cells of an `Hemitype` according to a list of `Chunk`.
| As for the `place` function, the `connect` method deals with connecting cells, and split
  the task into Chunks (by default as here, each chunk containing a presynaptic cell).

The parameters of `connect` are therefore the pre- and post-synaptic ``HemitypeCollection``.
This class provides a :meth:`placenent <.connectivity.strategy.HemitypeCollection.placement>`
method that we will use to iterate over its cell types populations ``PlacementSet``.

.. code-block:: python

    def connect(self, presyn_collection, postsyn_collection):
        # For each presynaptic placement set
        for pre_ps in presyn_collection.placement:
            # Load all presynaptic positions
            presyn_positions = pre_ps.load_positions()
            # For each postsynaptic placement set
            for post_ps in postsyn_collection.placement:
                # Load all postsynaptic positions
                postsyn_positions = post_ps.load_positions()

The next step is to filter the postsynaptic cells that are within the sphere of each of our
presynaptic cell. We can use the :doc:`norm <numpy:reference/generated/numpy.linalg.norm>`
function to measure the distance between one presynaptic cell and all its potential targets.
The ones we keep are within the ``radius`` defined as attribute of the class.

.. code-block:: python

    # For each presynaptic cell to connect
    for j, pre_position in enumerate(presyn_positions):
        # We measure the distance of each postsyn cell with respect to the
        # presyn cell
        dist = np.linalg.norm(postsyn_positions - pre_position, axis=1)
        # We keep only the ids that are within the sphere radius
        ids_to_keep = np.where(dist <= self.radius)[0]
        nb_connections = len(ids_to_keep)

Finally, we use the
:meth:`ConnectionStrategy.connect_cells <.connectivity.strategy.ConnectionStrategy.connect_cells>`
function, which will call assign a name to our `ConnectivitySet`.
This function requires for each individual pair of cell, their `connection location`:

- the index of the cell within its ``PlacementSet``
- the index of the morphology branch
- the index of the morphology branch point.

Because we are not using morphologies here the second and third indexes should be set to ``-1``:

.. code-block:: python

    for j, pre_position in enumerate(presyn_positions):
        # We measure the distance of each postsyn cell with respect to the
        # presyn cell
        dist = np.linalg.norm(postsyn_positions - pre_position, axis=1)
        # We keep only the ids that are within the sphere radius
        ids_to_keep = np.where(dist <= self.radius)[0]
        nb_connections = len(ids_to_keep)
        # We create two connection location array and set their neuron ids.
        pre_locs = np.full((nb_connections, 3), -1, dtype=int)
        pre_locs[:, 0] = j
        post_locs = np.full((nb_connections, 3), -1, dtype=int)
        post_locs[:, 0] = ids_to_keep

        self.connect_cells(pre_ps, post_ps, pre_locs, post_locs)

You have done it! Congrats! Your final `connectome.py` should look like this:

.. literalinclude:: /../examples/tutorials/dist_connection.py
    :language: python

.. tip::
    Comment your code! If not for you (because you are going to forget about it in a month),
    at least for the other people that will read it afterwards. |:wink:|

Enjoy
-----

You have done the hardest part! Now, you should be able to run the reconstruction once again
with your brand new components.

.. code-block:: bash

    bsb compile --verbosity 3

Remember that if you plan to create a bunch of components, it is best practice to keep your
component code in a subfolder with the same name as your model.
For example, if you are modelling the cerebellum, create a folder called ``cerebellum``.
Inside place an ``__init__.py`` file, so that Python can import code from it.

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1

    .. grid-item-card:: :octicon:`fold-up;1em;sd-text-warning` Make your own python project
        :link: projects
        :link-type: ref

        Learn how to configure your python project for BSB.

    .. grid-item-card:: :octicon:`tools;1em;sd-text-warning` Command-Line Interface
       :link: cli-guide
       :link-type: ref

       Familiarize yourself with BSB's CLI.

    .. grid-item-card:: :octicon:`gear;1em;sd-text-warning` Learn about components
       :link: main-components
       :link-type: ref

       Explore more about the main components.

    .. grid-item-card:: :octicon:`device-camera-video;1em;sd-text-warning` Examples
        :link: examples
        :link-type: ref

        Explore more advanced examples