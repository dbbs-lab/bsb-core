===================
Writing a component
===================

.. bsb_component_intro::

You can create custom connectivity patterns by creating a Python file in your project
root (e.g. ``my_module.py``) with inside a class inheriting from
:class:`~.connectivity.strategy.ConnectionStrategy`.

First we'll discuss the parts of the interface to implement, followed by an example, some
notes, and use cases.

Interface
---------

:meth:`~bsb.connectivity.strategy.ConnectionStrategy.connect`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``pre_set``/``post_set``: The pre/post-synaptic placement sets you used to perform the calculations.
* ``src_locs``/``dest_locs``:

   * A matrix with 3 columns with on each row the cell id, branch id, and
      point id.
   * Each row of the ``src_locs`` matrix will be connected to the same row in the
      ``dest_locs`` matrix

* ``tag`` : a tag describing the connection (optional, defaults to the strategy name, or
   `f"{name}_{pre}_to_{post}"` when multiple cell types are combined). Use this when you
   wish to create multiple distinct sets between the same cell types.

For example, if ``src_locs`` and ``dest_locs`` are the following matrices:

.. list-table:: src_locs
   :widths: 75 75 75
   :header-rows: 1

   * - Index of the cell in pre_pos array
     - Index of the branch at which the connection starts
     - Index of the point on the branch at which the connection starts.
   * - 2
     - 0
     - 6
   * - 10
     - 0
     - 2


.. list-table:: dest_locs
   :widths: 75 75 75
   :header-rows: 1

   * - Index of the cell in post_pos array
     - Index of the branch at which the connecion ends.
     - Index of the point on the branch at which the connection ends.
   * - 5
     - 1
     - 3
   * - 7
     - 1
     - 4

then two connections are formed:

* The first connection is formed between the presynaptic cell whose index in ``pre_pos``
   is ``2`` and the postsynaptic cell whose index in ``post_pos`` is ``10``.

Furthermore, the connection begins at the point with id ``6`` on the branch whose id is
   ``0`` on the presynaptic cell and ends on the points with id ``3`` on the branch whose
   id is ``1`` on the postsynaptic cell.

* The second connection is formed between the presynaptic cell whose index in ``pre_pos``
   is ``10`` and the postsynaptic cell whose index in ``post_pos`` is ``7``. Furthermore,
   the connection begins at the point with id ``3`` on the branch whose id is ``0`` on the
   presynaptic cell and ends on the points with id ``4`` on the branch whose id is ``1``
   on the postsynaptic cell.

.. note::
  If the exact location of a synaptic connection is not needed, then in both ``src_locs``
  and ``dest_locs`` the indices of the branches and of the point on the branch can be set
  to ``-1``.

:meth:`~bsb.connectivity.strategy.ConnectionStrategy.get_region_of_interest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an optional part of the interface. Using a region of interest (RoI) can speed
up algorithms when it is possible to know for a given presynaptic chunk, which
postsynaptic chunks might contain useful cell candidates.

Chunks are identified by a set of coordinates on a regular grid. E.g., for
a network with chunk size (100, 100, 100), the chunk (3, -2, 1) is the rhomboid region
between its least dominant corner at (300, -200, 100), and its most dominant corner at
(200, -100, 0).

``get_region_of_interest(chunk)`` receives the presynaptic chunk and should return a list
of postsynaptic chunks.

Example
-------

The example connects cells that are near each other, between a :guilabel:`min` and :guilabel:`max` distance:

.. code-block:: python

  from bsb.connectivity import ConnectionStrategy
  from bsb.exceptions import ConfigurationError
  from bsb import config
  import numpy as np
  import scipy.spatial.distance as dist

  @config.node
  class ConnectBetween(ConnectionStrategy):
    # Define the class' configuration attributes
    min = config.attr(type=float, default=0)
    max = config.attr(type=float, required=True)

    def connect(self, pre, post):
      # The `connect` function is responsible for deciding which cells get connected.
      # Use each hemitype's `.placement` to get a dictionary of `PlacementSet`s to connect

      # Cross-combine each presynaptic placement set ...
      for presyn_data in pre.placement:
        from_pos = presyn_data.load_positions()
        # ... with each postsynaptic placement set
        for postsyn_data in post.placement:
          to_pos = postsyn_data.load_positions()
          # Calculate the NxM pairwise distances between the cells
          pairw_dist = dist.cdist(from_pos, to_pos)
          # Find those that match the distance criteria
          m_pre, m_post = np.nonzero((pairw_dist <= max) & (pairw_dist >= min))
          # Construct the Kx3 connection matrices
          pre_locs = np.full((len(m_pre), 3), -1)
          post_locs = np.full((len(m_pre), 3), -1)
          # The first columns are the cell ids, the other columns are padded with -1
          # to ignore subcellular precision and form point neuron connections.
          pre_locs[:, 0] = m_pre
          post_locs[:, 0] = m_post
          # Call `self.connect_cells` to store the connections you found
          self.connect_cells(presyn_data, postsyn_data, pre_locs, post_locs)

    # Optional, you can leave this off to focus on `connect` first.
    def get_region_of_interest(self, chunk):
      # Find all postsynaptic chunks that are within the search radius away from us.
      return [
        c
        for c in self.get_all_post_chunks()
        if dist.euclidean(c.ldc, chunk.ldc) < self.max + chunk.dimensions
      ]

    # Optional, you can add extra checks and preparation of your component here
    def __init__(self, **kwargs):
      # Check if the configured max and min distance values make sense.
      if self.max < self.min:
        raise ConfigurationError("Max distance should be larger than min distance.")

And an example configuration using this strategy:

.. code-block:: json

  {
    "components": ["my_module.py"],
    "connectivity": {
      "type_A_to_type_B": {
        "class": "my_module.ConnectBetween",
        "min": 10,
        "max": 15.5,
        "presynaptic": {
          "cell_types": ["type_A"]
        },
        "postsynaptic": {
          "cell_types": ["type_B"]
        }
      }
    }
  }

Notes
~~~~~

.. rubric:: Setting up the class

We need to inherit from :class:`~bsb.connectivity.strategy.ConnectionStrategy` to create a
connection component and decorate our class with the ``config.node`` decorator to
integrate it with the configuration system. For specifics on configuration, see
:doc:`/config/nodes`.

.. rubric:: Accessing configuration values during ``connect``

Any ``config.attr`` or similar attributes that you define on the class will be populated
with data from the network configuration, and will be available on ``self`` in the
methods of the component.

In this example :guilabel:`min` is an optional float that defaults to 0, and
:guilabel:`max` is a required float.

.. rubric:: Accessing placement data during ``connect``

The ``connect`` function is handed the placement information as the ``pre`` and ``post``
parameters. The ``.placement`` attribute contains a dictionary with as keys the
:class:`.cell_types.CellType` and as value the
:class:`PlacementSets <.storage.interfaces.PlacementSet>`.

.. note::
  The placement sets in the parameters are scoped to the data of the parallel job that is
  being executed. If you want to remove this scope and access to the global data, you can
  create a fresh placement set from the cell type with ``cell_type.get_placement_set()``.

.. rubric:: Creating connections

Connections are stored in a presynaptic and postsynaptic matrix. Each matrix contains 3
columns: the cell id, branch id, and point id. If your cells have no morphologies, use -1
as a filler for the branch and point ids.

Call ``self.scaffold.connect_cells(from_type, to_type, from_locs, to_locs)`` to connect
the cells. If you are creating multiple different connections between the same pair of cell
types, you can pass an optional ``tag`` keyword argument to give them a unique name and
separate them.

.. rubric:: Use regions of interest

Using a region of interest (RoI) can speed up algorithms when it is possible to know,
when given a presynaptic chunk, which postsynaptic chunks might contain useful cell
candidates.

Chunks are identified by a set of coordinates on a regular grid. E.g., for
a network with chunk size (100, 100, 100), the chunk (3, -2, 1) is the rhomboid region
between its least dominant corner at (300, -200, 100), and its most dominant corner at
(200, -100, 0).

Using the same example, for every presynaptic chunk, we know that we will only form
connections with cells less than ``max`` distance away, so why check cells in chunks more
than ``max`` distance away?

If you implement ``get_region_of_interest(chunk)``, you can return the list of chunks that
should be loaded for the parallel job that processes that ``chunk``:

.. code-block:: python

  def get_region_of_interest(self, chunk):
    return [
      c
      for c in self.get_all_post_chunks()
      if dist.euclidean(c.ldc, chunk.ldc) < self.max + chunk.dimensions
    ]

Connecting point-like cells
---------------------------
Suppose we want to connect Golgi cells and granule cells, without storing information
about the exact positions of the synapses (we may want to consider cells as point-like
objects, as in NEST). We want to write a class called ``ConnectomeGolgiGranule`` that
connects a Golgi cell to a granule cell if their distance is less than 100 micrometers
(see the configuration block above).

First we define the class ``ConnectomeGolgiGlomerulus`` and we specify that we require
to be configured with a :guilabel:`radius` and :guilabel:`divergence` attribute.

.. code-block:: python

  @config.node
  class ConnectomeGolgiGlomerulus(ConnectionStrategy):
      # Read vars from the configuration file
      radius = config.attr(type=int, required=True)
      divergence = config.attr(type=int, required=True)

Now we need to write the ``get_region_of_interest`` method. For a given chunk we want
all the neighbouring chunks in which we can find the presynaptic cells at less than 50
micrometers. Such cells are contained for sure in the chunks which are less than 50
micrometers away from the current chunk.

.. code-block:: python

    def get_region_of_interest(self, chunk):
      # We get the ConnectivitySet of golgi_to_granule
      cs = self.network.get_connectivity_set(tag="golgi_to_granule")
      # We get the coordinates of all the chunks
      chunks = ct.get_placement_set().get_all_chunks()
      # We define an empty list in which we shall add the chunks of interest
      selected_chunks = []
      # We look for chunks which are less than radius away from the current one
      for c in chunks:
        dist = np.sqrt(
          np.power((chunk[0] - c[0]) * chunk.dimensions[0], 2)
            + np.power((chunk[1]  - c[1]) * chunk.dimensions[1], 2)
            + np.power((chunk[2]  - c[2]) * chunk.dimensions[2], 2)
        )
        # We select only the chunks satisfying the condition
        if (dist < self.radius):
            selected_chunks.append(Chunk([c[0], c[1], c[2]], chunk.dimensions))
      return selected_chunks

Now we're ready to write the ``connect`` method:

.. code-block:: python

    def connect(self, pre, post):
      # This strategy connects every combination pair of the configured presynaptic to postsynaptic cell types.
      # We will tackle each pair's connectivity inside of our own `_connect_type` helper method.
      for pre_ps in pre.placement:
          for post_ps in post.placement:
              # The hemitype collection's `placement` is a dictionary mapping each cell type to a placement set with all
              # cells being processed in this parallel job. So call our own `_connect_type` method with each pre-post combination
              self._connect_type(pre_ps, post_ps)

      def _connect_type(self, pre_ps, post_ps):
        # This is the inner function that calculates the connectivity matrix for a pre-post cell type pair
        # We start by loading the cell position matrices (Nx3)
        golgi_pos = pre_ps.load_positions()
        granule_pos = post_ps.load_positions()
        n_glomeruli = len(glomeruli_pos)
        n_golgi = len(golgi_pos)
        n_conn = n_glomeruli * n_golgi
        # For the sake of speed we define two arrays pre_locs and post_locs of length n_conn
        # (the maximum number of connections which can be made) to store the connections information,
        # even if we will not use all the entries of arrays.
        # We keep track of how many entries we actually employ, namely how many connection
        # we made, using the variable ptr. For example if we formed 4 connections the useful
        # data lie in the first 4 elements
        pre_locs = np.full((n_conn, 3), -1, dtype=int)
        post_locs = np.full((n_conn, 3), -1, dtype=int)
        ptr = 0
        # We select the cells to connect according to our connection rule.
        for i, golgi in enumerate(golgi_pos):
          # We compute the distance between the current Golgi cell and all the granule cells in the region of interest.
          dist = np.sqrt(
                      np.power(golgi[0] - granule_pos[0], 2)
                      + np.power(golgi[1] - granule_pos[1], 2)
                      + np.power(golgi[2] - granule_pos[2], 2)
                  )
          # We select all the granule cells which are less than 100 micrometers away up to the divergence value.
          # For the sake of simplicity in this example we assume to find at least 40 candidates satisfying the condition.
          granule_close_enough = dist < self.radius

          # We find the indices of the 40 closest granule cells
          to_connect_ids = np.argsort(granule_close_enough)[0:self.divergence]

          # Since we are interested in connecting point-like cells, we do not need to store
          # info about the precise position on the dendrites or axons;
          # It is enough to store which presynaptic cell is connected to
          # certain postsynaptic cells, namely the first entry of both `pre_set` and `post_set`.

          # The index of the presynaptic cell in the `golgi_pos` array is `i`
          pre_set[ptr:ptr+self.divergence,0] = i
          # We store in post_set the indices of the postsynaptic cells we selected before.
          post_set[ptr:ptr+self.divergence,0] = to_connect_ids
          ptr += to_be_connected

        # Now we connect the cells according to the information stored in `src_locs` and `dest_locs`
        # calling the `connect_cells` method.
        self.connect_cells(pre_set, post_set, src_locs, dest_locs)

Connections between a detailed cell and a point-like cell
---------------------------------------------------------

If we have a detailed morphology of the pre- or postsynaptic cells we can specify where
to form the connection. Suppose we want to connect Golgi cells to glomeruli specifying
the position of the connection on the Golgi cell axon. In this example we form a
connection on the closest point to a glomerulus. First, we need to specify the type of
neurites that we want to consider on the morphologies when forming synapses. We can do
this in the configuration file, using the:guilabel:`morphology_labels` attribute on the
``connectivity.*.postsynaptic`` (or ``presynaptic``) node:

.. code-block:: json

  "golgi_to_granule": {
        "strategy": "cerebellum.connectome.golgi_granule.ConnectomeGolgiGranule",
        "radius": 100,
        "convergence": 40,
        "presynaptic": {
          "cell_types": ["glomerulus"]
        },
        "postsynaptic": {
          "cell_types": ["golgi_cell"],
          "morphology_labels" : ["basal_dendrites"]
        }
      }

The :meth:`~bsb.connectivity.strategy.ConnectionStrategy.get_region_of_interest` is
analogous to the previous example, so we focus only on the
:meth:`~bsb.connectivity.strategy.ConnectionStrategy.connect` method.

.. code-block:: python

    def connect(self, pre, post):
      for pre_ps in pre.placement:
          for post_ps in post.placement:
              self._connect_type(pre_ps, post_ps)

      def _connect_type(self, pre_ps, post_ps):
        # We store the positions of the pre and post synaptic cells.
        golgi_pos = pre_ps.load_positions()
        glomeruli_pos = post_ps.load_positions()
        n_glomeruli = len(glomeruli_pos)
        n_golgi = len(golgi_pos)
        max_conn = n_glomeruli * n_golgi
        # We define two arrays of length `max_conn ` to store the connections,
        # even if we will not use all the entries of arrays, for the sake of speed.
        pre_locs = np.full((max_conn , 3), -1, dtype=int)
        post_locs = np.full((max_conn , 3), -1, dtype=int)
        # `ptr` keeps track of how many connections we've made so far.
        ptr = 0

        # Cache morphologies and generate the morphologies iterator.
        morpho_set = post_ps.load_morphologies()
        golgi_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        # Loop through all the Golgi cells
        for i, golgi, morpho in zip(itertools.count(), golgi_pos, golgi_morphos):

            # We compute the distance between the current Golgi cell and all the glomeruli,
            # then select the good ones.
            dist = np.sqrt(
                np.power(golgi[0] - glomeruli_pos[:, 0], 2)
                + np.power(golgi[1] - glomeruli_pos[:, 1], 2)
                + np.power(golgi[2] - glomeruli_pos[:, 2], 2)
            )

            to_connect_bool = dist < self.radius
            to_connect_idx = np.nonzero(to_connect_bool)[0]
            connected_gloms = len(to_connect_idx)

            # We assign the indices of the Golgi cell and the granule cells to connect
            pre_locs[ptr : (ptr + connected_gloms), 0] = to_connect_idx
            post_locs[ptr : (ptr + connected_gloms), 0] = i

            # Get the branches corresponding to basal dendrites.
            # `morpho` contains only the branches tagged as specified
            # in the configuration file.
            basal_dendrides_branches = morpho.get_branches()

            # Get the starting branch id of the denridic branches
            first_dendride_id = morpho.branches.index(basal_dendrides_branches[0])

            # Find terminal points on branches
            terminal_ids = np.full(len(basal_dendrides_branches), 0, dtype=int)
            for i,b in enumerate(basal_dendrides_branches):
                if b.is_terminal:
                    terminal_ids[i] = 1
            terminal_branches_ids = np.nonzero(terminal_ids)[0]

            # Keep only terminal branches
            basal_dendrides_branches = np.take(basal_dendrides_branches, terminal_branches_ids, axis=0)
            terminal_branches_ids = terminal_branches_ids + first_dendride_id

            # Find the point-on-branch ids of the tips
            tips_coordinates = np.full((len(basal_dendrides_branches),3), 0, dtype=float)
            for i,branch in enumerate(basal_dendrides_branches):
                tips_coordinates[i] = branch.points[-1]

            # Choose randomly the branch where the synapse is made
            # favouring the branches closer to the glomerulus.
            rolls = exp_dist.rvs(size=len(basal_dendrides_branches))

            # Compute the distance between terminal points of basal dendrites
            # and the soma of the avaiable glomeruli
            for id_g,glom_p in enumerate(glomeruli_pos):
                pts_dist = np.sqrt(np.power(tips_coordinates[:,0] + golgi[0] - glom_p[0], 2)
                        + np.power(tips_coordinates[:,1] + golgi[1] - glom_p[1], 2)
                        + np.power(tips_coordinates[:,2] + golgi[2] - glom_p[2], 2)
                    )

                sorted_pts_ids = np.argsort(pts_dist)
                # Pick the point in which we form a synapse according to a exponential distribution mapped
                # through the distance indices: high chance to pick closeby points.
                pt_idx = sorted_pts_ids[int(len(basal_dendrides_branches)*rolls[np.random.randint(0,len(rolls))])]

                # The id of the branch is the id of the terminal_branches plus the id of the first dendritic branch
                post_locs[ptr+id_g,1] = terminal_branches_ids[pt_idx]
                # We connect the tip of the branch
                post_locs[ptr+id_g,2] = len(basal_dendrides_branches[pt_idx].points)-1
            ptr += connected_gloms

        # Now we connect the cells
        self.connect_cells(pre_ps, post_ps, pre_locs[:ptr], post_locs[:ptr])

