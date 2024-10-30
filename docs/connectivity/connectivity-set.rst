#################
Connectivity sets
#################

The :class:`ConnectivitySet <.storage.interfaces.ConnectivitySet>` represents the
ensemble of all connections established in your network.
It is constructed from the :class:`~.storage.Storage` and can be used to retrieve
information about pre- and post-cell types, as well as the connections between them.

Retrieving a ConnectivitySet
============================

It is possible to list all the ``ConnectivitySet`` stored in your scaffold, here,
loaded from the network file ``"my_network.hdf5"``:

.. code-block:: python

  from bsb import from_storage

  scaffold = from_storage("my_network.hdf5")

  cs_list = scaffold.get_connectivity_sets()
  for cs in cs_list:
    print (f"ConnectivitySet {cs.tag} has {len(cs)} connections")

Alternatively, is possible to load the set by its name:

.. code-block:: python

     cs = scaffold.get_connectivity_set("my_CS")

The cell type information is accessible through the attributes :guilabel:`pre_type` and :guilabel:`post_type`:

.. code-block:: python

    # Get the pre-/post-synaptic cell types
    ct_pre = cs.pre_type
    ct_pos = cs.pos_type

    # Get the name of the pre-/post-synaptic cell types
    print(f"My pre-type is {cs.pre_type_name}")
    print(f"My post-type is {cs.post_type_name}")

Connections
===========

A list of all the cell pairs for every ``ConnectivitySet`` can be loaded with the
method :meth:`~.storage.interfaces.ConnectivitySet.load_connections`.

.. code-block:: python

    for (src_locs, dest_locs) in cs.load_connections():
        print(f"Cell id: {src_locs[0]} connects to cell {dest_locs[0]}")

Here, ``src_locs`` and ``dest_locs`` contains for each ``pair``:

- the cell id
- the cell morphology branch id
- the cell morphology branch section id

.. note::
    If the pre-/post-synaptic neuron does not have a morphology then
    their branch and section id in the connection is ``-1``