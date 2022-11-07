#####################
Connectivity strategy
#####################

:light-bulb: Make sure to read the :ref:`Writing Components <components>` section first.

:class:`~bsb.connectivity.strategy.ConnectionStrategy` is an abstract class allowing to define a connection strategy between cells.
All the data about connection strategies are in the connectivity section of the :ref:`Configuration file <files>`.
For example suppose we are writing a cerebellum module and that we are defining a connection strategy ConnectomeGolgiGranule to connect Golgi cells to granule cells in golgi_granule.py.
Suppose that up to 40 granule cells can be connected to a Golgi cell and that a connection can be formed only if the somata of two cells are less than 100 micrometers away. 
In the case in which more than 40 cells meet the conditions, we take the 40 closer ones.

.. code-block:: json
  
  "golgi_to_granule": {
        "strategy": "cerebellum.connectome.golgi_granule.ConnectomeGolgiGranule",
        "radius": 100,
        "convergence": 40,
        "presynaptic": {
          "cell_types": ["golgi_cell"]
        },
        "postsynaptic": {
          "cell_types": ["granule_cell"]
        }
      }

The prototype of a custom connection strategy is the following:

.. code-block:: python

  #We import the base class ConnectionStrategy
  from bsb.strategy import ConnectionStrategy
  #We import config to read the variables from the configuration file
  from bsb import config
  #We import numpy because we shall use it to perform the math operations 
  #need to select the cells to connect.
  import numpy as np

  class ConnectomeGolgiGranule(ConnectionStrategy):
    
    def get_region_of_interest(self, chunk):
      #For a given chunk this method returns a list of chunks in which to look for postsynaptic cells. 

    def connect(self, pre, post):
      #Here goes the code that selects the cells to connect
      #The information about the connections to be formed are stored
      #in two matrices to be passed to the connect connect_cells method,
      #to be called at the end. 

.. note::
  Due to performance and memory reasons, the connections are not formed processing the whole simulation volume at once, since it would require a lot of memory, time and computational power. Instead, the volume is divided in chunks, which may be processed in parallel to further speed up the creation of the connectome, and the connections are formed on a chunk by chunk basis. This step is handles by the :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.queue` method of the base class :class:`~bsb.connectivity.strategy.ConnectionStrategy` : The user writing a custom connection strategy does not need to care about the subdivision in chucks, since it is handled automatically by the framework. 

.. note::
  By default a single presyptic-chunk is associated with many post-synaptic chunks individuated by get_region_of_interest. Therefore, the argument post contains the data about the postsynaptic cells in the chunks inside the region of interest (ROI), while pre contains data about the presynaptic cells of a single chunk. 
  However, when writing a custom connection strategy it may be useful to do the opposite, namely to associate a single post-synaptic chunk to many pre-synaptic chunks, an example being the connection between mossy fibers and glomeruli, for which we need to make sure that each glomerulus is associated to one and only one mossy fiber. 
  This can be done overriding the function :meth:`~.bsb.connectivity.strategy.ConnectionStrategy._get_connect_args_from_job`, from which the arguments pre and post of the connect method connect come from. 
  To do so, it is enough to add the following code to the prototype above.

  .. code-block:: python
  
    def _get_connect_args_from_job(self, chunk, roi):
          pre = HemitypeCollection(self.presynaptic, roi)
          post = HemitypeCollection(self.postsynaptic, [chunk])
          return pre, post

get_region_of_interest method
-----------------------------

Arguments: a chunk containing the postsynaptic cells.
The goal of this method is to find all the chunks in the simulation volume containing all the possibile the postsynaptic cells of the presynaptic cells in the chunk given as argument.

connect method
--------------

Arguments: ``pre`` and ``post`` are ConnectionCollections. By default post contains the data about the postsynaptic cells in the region of interest individuated by  :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.get_region_of_interest`.

.. note::
  The user does not need to call :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.get_region_of_interest` inside the connect method, since it's automatically called in the :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.queue` method of the base class :class:`~bsb.connectivity.strategy.ConnectionStrategy`.

The connection between two types of cells is made calling the connect_cells method.
connect_cells needs four arguments: 
 * ``pre_set`` : A numpy array containing the positions of the presynaptic cells.
 * ``post_set`` : A numpy array containing the positions of the postsynaptic cells.
 * ``src_locs`` : A nx3 matrix, with n the number of connections, containing information about where the connection starts.
Each row of the matrix contains three integers (a,b,c), with a the index of the presynaptic cell, b the index of the branch on which a connection is made 
and c the index (relative to a branch) of the point at which the connection starts.  
  * ``dest_locs`` : A nx3 matrix,with n the number of connections, containing information about where the connection ends.
Each row of the matrix contains three integers (a,b,c), with a the index of the postsynaptic cell, b the index of the branch on which a connection is made 
and c the index (relative to a branch) of the point at which the connection ends. 
The k-th row of src_locs describes the beginning of the k-th connection on the presynaptic cell, while the k-th row of dest_locs stores the info about the end of the k-th connection on the postsynaptic cell. 
There is also an optional argument: 
 * ``tag`` : a tag describing the connection. 

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

* The first connection is formed between the presynaptic cell whose index in pre_pos is 2 and the postsynaptic cell whose index in post_pos is 10.
Furthermore, the connection begins at the point with id 6 on the branch whose id is 0 on the presynaptic cell and ends on the points with id 3 on the branch whose id is 1 on the postsynaptic cell.
* The second connection is formed between the presynaptic cell whose index in pre_pos is 10 and the postsynaptic cell whose index in post_pos is 7.
Furthermore, the connection begins at the point with id 3 on the branch whose id is 0 on the presynaptic cell and ends on the points with id 4 on the branch whose id is 1 on the postsynaptic cell. 

.. note::
  If the exact location of a synaptic connection is not needed, then in both src_locs and dest_locs the indices of the branches and of the point on the branch can be set to -1.

Use case 1 : Connect point-like cells 
=====================================
Suppose we want to connect Golgi cells and granule cells, without storing information about the exact positions of the synapses (we may want to consider cells as point-like objects, as in NEST).
We want to write a class called ConnectomeGolgiGranule that connects a Golgi cell to a granule cell if their distance is less than 100 micrometers, see the configuration block above. 

First we define the class ConnectomeGolgiGlomerulus and we read the radius and the divergence target.

.. code-block:: python

  @config.node
  class ConnectomeGolgiGlomerulus(ConnectionStrategy):
      # Read vars from the configuration file
      radius = config.attr(type=int, required=True)
      divergence = config.attr(type=int, required=True)

Now we need to write the get_region_of_interest method.
For a given chunk we want all the neighbouring chunks in which we can find the presynaptic cells at less than 50 micrometers.
Such cells are contained for sure in the chunks which are less than 50 micrometers away from the current chunk.

.. code-block:: python

    def get_region_of_interest(self, chunk):
      #We get the ConnectivitySet of golgi_to_granule
      cs = self.network.get_connectivity_set(tag="golgi_to_granule")
      #We get the coordinates of all the chunks
      chunks = ct.get_placement_set().get_all_chunks()
      #We define an empty list in which we shall add the chunks of interest
      selected_chunks = []
        #We look for chunks which are less than radius away from the current one
        for c in chunks:    
            dist = np.sqrt(
                np.power((chunk[0] - c[0]) * chunk.dimensions[0], 2)
                + np.power((chunk[1]  - c[1]) * chunk.dimensions[1], 2)
                + np.power((chunk[2]  - c[2]) * chunk.dimensions[2], 2)
            )
            #We select only the chunks satisfying the condition
            if (dist < self.radius):
                selected_chunks.append(Chunk([c[0], c[1], c[2]], chunk.dimensions))
        return selected_chunks
    
Finally we are ready to write the connect method. 

.. code-block:: python

    def connect(self, pre, post):
      #We extract information about the pre and post synaptic cells
      pre_type = pre.cell_types[0]
      post_type = post.cell_types[0]
      for pre_ct, pre_ps in pre.placement.items():
          for post_ct, post_ps in post.placement.items():
              #We select the cells to connect and we connect them.
              #Here pre_ps contains only the positions of the presynaptic cells in the region of interest
              #and post_ps contains only the positions of the postsynaptic cells in the chunk currently being processed.
              self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

      def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        #We get the positions of the pre and post synaptic cells.
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
        #We select the cells to connect according to our connection rule.
        for i, golgi in enumerate(golgi_pos):
          #We compute the distance between the current Golgi cell and all the granule cells in the region of interest.
          dist = np.sqrt(
                      np.power(golgi[0] - granule_pos[0], 2)
                      + np.power(golgi[1] - granule_pos[1], 2)
                      + np.power(golgi[2] - granule_pos[2], 2)
                  )
          #We select all the granule cells which are less than 100 micrometers away up to the divergence value.
          #For the sake of simplicity in this example we assume to find at least 40 candidates satisfying the condition.
          granule_close_enough = dist < self.radius
          
          #We find the indices of the 40 closest granule cells
          to_connect_ids = np.argsort(granule_close_enough)[0:self.divergence]

          #Since we are interested in connecting point-like cells, we do not need to store
          #info about the precise position on the dendrites or axons; 
          #It is enough to store which presynaptic cell is connetcted to
          #certain postsynaptic cells, namely the first entry of both pre_set and post_set.
          
          #The index of the presynaptic cell in the golgi_pos array is i
          pre_set[ptr:ptr+self.divergence,0] = i
          #We store in post_set the indices of the postsynaptic cells we selected before.
          post_set[ptr:ptr+self.divergence,0] = to_connect_ids
          ptr += to_be_connected

        #Now we connect the cells according to the information stored in src_locs and dest_locs
        #calling the connect_cells method.
        connect_cells(pre_set, post_set, src_locs, dest_locs)

Use case 2 : Connection between a detailed cell and a point-like cell. 
======================================================================

If we have a detailed morphology of the pre or post synaptic cells we can specify where to form the connection. Suppose we want to connect Golgi cells to glomeruli specifying the position of the connection on the Golgi cell axon. In this example we form a connection on the closest point to a glomerulus.
First, we need to specify the neurites where we want to form a synapse in the configuration file, using the "morphology_labels" tag. 

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

The :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.get_region_of_interest` is analogous to the previous example, so we focus only on the :meth:`~.bsb.connectivity.strategy.ConnectionStrategy.connect` method.

.. code-block:: python

    def connect(self, pre, post):
      #We extract information about the pre and post synaptic cells
      pre_type = pre.cell_types[0]
      post_type = post.cell_types[0]
      for pre_ct, pre_ps in pre.placement.items():
          for post_ct, post_ps in post.placement.items():
              #We select the cells to connect and we connect them.
              #Here pre_ps contains only the positions of the presynaptic cells in the region of interest
              #and post_ps contains only the positions of the postsynaptic cells in the chunk currently being processed.
              self._connect_type(pre_ct, pre_ps, post_ct, post_ps)
  
      def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        #We store the positions of the pre and post synaptic cells.
        golgi_pos = pre_ps.load_positions()
        glomeruli_pos = post_ps.load_positions()
        n_glomeruli = len(glomeruli_pos)
        n_golgi = len(golgi_pos)
        n_conn = n_glomeruli * n_golgi
        # We define two arrays of length n_conn to store the connections to be made,
        # even if we will not use all the entries of arrays, for the sake of speed.
        # We keep track of how many entries, namely how many connection, by means of the variable ptr.
        pre_locs = np.full((n_conn, 3), -1, dtype=int)
        post_locs = np.full((n_conn, 3), -1, dtype=int)
        ptr = 0
      
        #Cache morphologies and generate the morphologies iterator.
        morpho_set = post_ps.load_morphologies()
        golgi_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)
        
        #Loop through all the Golgi cells
        for i, golgi, morpho in zip(itertools.count(), golgi_pos, golgi_morphos):
            
            #We compute the distance between the current Golgi cell and all the glomeruli,
            #then select the good ones.
            dist = np.sqrt(
                np.power(golgi[0] - glomeruli_pos[:, 0], 2)
                + np.power(golgi[1] - glomeruli_pos[:, 1], 2)
                + np.power(golgi[2] - glomeruli_pos[:, 2], 2)
            )
            
            to_connect_bool = dist < self.radius
            to_connect_idx = np.nonzero(to_connect_bool)[0]
            connected_gloms = len(to_connect_idx)
            
            #We assign the indices of the Golgi cell and the granule cells to connect 
            pre_locs[ptr : (ptr + connected_gloms), 0] = to_connect_idx
            post_locs[ptr : (ptr + connected_gloms), 0] = i
           
            #Get the branches corresponding to basal dendrites.
            #morpho contains only the branches tagged as specified 
            #in the configuration file.
            basal_dendrides_branches = morpho.get_branches()
            
            #Get the starting branch id of the denridic branches
            first_dendride_id = morpho.branches.index(basal_dendrides_branches[0])
            
            #Find terminal points on branches
            terminal_ids = np.full(len(basal_dendrides_branches), 0, dtype=int)
            for i,b in enumerate(basal_dendrides_branches):
                if b.is_terminal:
                    terminal_ids[i] = 1
            terminal_branches_ids = np.nonzero(terminal_ids)[0]

            #Keep only terminal branches
            basal_dendrides_branches = np.take(basal_dendrides_branches, terminal_branches_ids, axis=0)
            terminal_branches_ids = terminal_branches_ids + first_dendride_id

            #Find the point-on-branch ids of the tips
            tips_coordinates = np.full((len(basal_dendrides_branches),3), 0, dtype=float)
            for i,branch in enumerate(basal_dendrides_branches):
                tips_coordinates[i] = branch.points[-1]

            #Choose randomly the branch where the synapse is made
            #favouring the branches closer to the glomerulus.
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

                #The id of the branch is the id of the terminal_branches plus the id of the first dendritic branch
                post_locs[ptr+id_g,1] = terminal_branches_ids[pt_idx]
                #We connect the tip of the branch
                post_locs[ptr+id_g,2] = len(basal_dendrides_branches[pt_idx].points)-1
            ptr += connected_gloms

        #Now we connect the cells
        self.connect_cells(pre_ps, post_ps, pre_locs[:ptr], post_locs[:ptr])

