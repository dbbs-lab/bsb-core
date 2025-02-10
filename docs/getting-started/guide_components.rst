.. _guide_components:

Create your own components
==========================

.. note::
    This guide assumes that you have a good understanding of the basis of BSB; i.e., that
    you are familiar with the concept of BSB ``components``. You should have read the whole
    the getting started :doc:`section </getting-started/top-level-guide>`.

In this tutorial, we are going to guide you through the process of creating your own
component for a new :doc:`Connection strategy </connectivity/connection-strategies>`.
We will start from the following configuration file (corresponds to the first network file
from the getting started tutorial):

.. literalinclude:: /getting-started/configs/getting-started.json
    :language: json

Let's save this new configuration in our project folder under the name ``config_connectome.json``

Description of the strategy to implement
----------------------------------------

For the connectivity, we will consider that the cells ``base_type`` can connect to all ``top_type`` cells
within a sphere of ``radius`` ``100`` µm. You can consider this as a simplified model of
distance based connectivity.

Components boiler plate
-----------------------

BSB components are written as Python classes. When the BSB parses your `Configuration`
component, it resolves the path to its class or function and import it. Hence, your
components should be written as
`importable modules <https://docs.python.org/3/tutorial/modules.html>`_.

In our case, we will place our `ConnectionStrategy` class in ``connectome.py``:

.. code-block:: python

    from bsb import config, ConnectionStrategy

    @config.node
    class DistanceConnectivity(ConnectionStrategy):

        # add your attributes here

        def connect(self, presyn_collection, postsyn_collection):
            # write your code here
            pass

The :class:`ConnectionStrategy <.connectivity.strategy.ConnectionStrategy>` here requires you to
implement the :meth:`connect <.connectivity.strategy.ConnectionStrategy.connect>` function.

.. tip::
    Take the time to familiarize yourself with the class and the function before continuing.

.. note::
    In this example, we only implemented the required function for the strategy but
    you can also overwrite the other functions of the interface, if you need it.
    Please refer to the documentation on the classes you want to implement for more
    information.

Note that this strategy leverages the ``@config.node``
`python decorator <https://pythonbasics.org/decorators/>`_.
The :doc:`configuration node decorator</config/nodes>` allows you to pass the parameters
defined in the configuration file to the class. It will also handle the
:doc:`type testing </config/types>` of your configuration attributes (e.g., make sure your
``radius`` parameter is a positive float number). We will see in the following sections how
to create your class configuration attributes.

Add configuration attributes
----------------------------

For our strategy, we need to pass a list of parameters to its class,
through our configuration file.

Here, we need a radius parameter which translates into the following code in our class:

.. code-block:: python

    from bsb import config, ConnectionStrategy, types

    @config.node
    class DistanceConnectivity(ConnectionStrategy):

        radius: float = config.attr(type=types.float(min=0), required=True)

        def connect(self, presyn_collection, postsyn_collection):
            # write your code here
            pass

Here, :guilabel:`radius` is a positive float that is required, this means that the BSB will throw a
:class:`ConfigurationError <.exceptions.ConfigurationError>` if the parameter is not provided.
This will also happen if the parameters provided for the configuration attributes do not match
the expected types.

At this stage, you have created a python class with minimal code implementation, you should
now link it to your configuration file. To import our class in our configuration file, we
will modify the :guilabel:`connectivity` block:

.. code-block:: json

  "connectivity": {
    "A_to_B": {
      "strategy": "connectome.DistanceConnectivity",
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

Here, we are going to use the `connect` function to produce and store
:class:`ConnectivitySets <.storage.interfaces.ConnectivitySet>`.
First some definition:

| The presynaptic and postsynaptic populations to connect (each can have multiple cell type
  populations) are called :class:`Hemitype <.connectivity.strategy.Hemitype>`. An `Hemitype`
  serves as the interface to define a connection population and its parameters in the
  Configuration.
| The class :class:`HemitypeCollection <.connectivity.strategy.HemitypeCollection>` allows
  you to filter the cells of an `Hemitype` according to a list of `Chunk`.
| The `connect` method deals with connecting cells, and split the task into Chunks
  (here, each chunk containing a presynaptic cell).

The parameters of `connect` are therefore the pre- and post-synaptic ``HemitypeCollection``.
This class provides a :meth:`placement <.connectivity.strategy.HemitypeCollection.placement>`
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
function, which will create and store our resulting `ConnectivitySet`. It will also assign it a name
based on the Strategy name and eventually the pre- and post-synaptic populations connected (if there
are more than one pair).
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
with your brand new component.

.. code-block:: bash

    bsb compile config_connectome.json --verbosity 3

It is best practice to keep your component code in a subfolder with the same name as
your model. For example, if you are modelling the cerebellum, create a folder called
``cerebellum``. Inside place an ``__init__.py`` file, so that Python can import code from
it. Then you best subdivide your code based on component type, e.g. keep connectivity
strategies in a file called ``connectome.py``. That way, your connectivity components are
available in your model as ``cerebellum.connectome.MyComponent``. It will also make it
easy to distribute your code as a package!

More advanced component writing
-------------------------------
If you want to see another example on how to write BSB components, you can take a look at
the placement strategy example in :doc:`this section </examples/place_distribution>`

.. rubric:: Next steps:

.. grid:: 1 1 1 2
    :gutter: 1

    .. grid-item-card:: :octicon:`fold-up;1em;sd-text-warning` Start contributing!
        :link: development-section
        :link-type: ref

        Help out the project by contributing code.

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