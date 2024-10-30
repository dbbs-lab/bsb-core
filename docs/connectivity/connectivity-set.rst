#################
Connectivity sets
#################

The :class:`ConnectivitySet <.storage.interfaces.ConnectivitySet>` represents the
ensemble of all connections established in your network.
It is constructed from the :class:`~.storage.Storage` and can be used to retrieve
information about pre- and post-cell types, as well as the connections between them.

Retrieving a ConnectivitySet
============================

It is possible to retrieve the ``ConnectivitySet`` stored in our network, for example,
from the network file "my_network.hdf5":

.. code-block:: python

  from bsb import from_storage
  network = from_storage("my_network.hdf5")

  cs_list = network.get_connectivity_sets()
  for cs in cs_list:                            #  Print the name of all
    print ("ConnectivitySet found:",cs.tag)     #  the ConnectivitySets stored

Alternatively, is possible to load the set by its name:

.. code-block:: python

     cs = network.get_connectivity_set("my_CS")

The cell type information is accessible through the attributes :guilabel:`pre_type` and :guilabel:`post_type`:

.. code-block:: python

    ct_pre = cs.pre_type   # Load the pre-synaptic cell type
    ct_pos = cs.pos_type   # Load the post-synaptic cell type

    print("My pre-type is", cs.pre_type_name)   # Access the name of the pre-synaptic cell type
    print("My post-type is", cs.post_type_name)  # Access the name of the post-synaptic cell type

Connections
===========

A list of all the cell pairs for every ``ConnectivitySet`` can be loaded with the
method :meth:`~.storage.interfaces.ConnectivitySet.load_connections`.

.. code-block:: python

    connection_pairs = []

    for pair in cs.load_connections():
        connection_pairs.append(pair)

