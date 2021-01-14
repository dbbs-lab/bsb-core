==============
Plotting Tools
==============

The scaffold package provides tools to plot network topology (either point and detailed
networks) and morphologies in the ``bsb.plotting`` module.

To plot a network saved in a `bsb` instance, you can use:

* ``plot_network_cache(scaffold)``: to plot the network saved in the memory cache after
  having compiled it
* ``plot_network(scaffold)``: to plot a network, adding the keyword argument
  ``from_memory=False`` if you want to plot a network saved in a previously compiled HDF5
  file. The default value is ``from_memory=True``, which plots the version saved in your
  cache (you should have compiled the network in the current session).
* ``plot_network_detailed(scaffold)``: Plots cells represented by their fully detailed
  morphologies. These plots are usually not able to render more than a 30-50 cells at the
  same time depending on the complexity of their morphology.


You can also plot morphologies:

* ``plot_morphology(m)``: Plots a :class:`Morphology <.morphologies.Morphology>`
* ``plot_fiber_morphology(fm)``: Plots a :class:`FiberMorphology <.networks.FiberMorphology>`
* ``plot_voxel_cloud(m.cloud)``: Plots a :class:`VoxelCloud <.voxels.VoxelCloud>`

All of the above functions take a ``fig`` keyword argument of type
:class:`plotly.graph_objects.Figure` in case you want to modify the figure, or combine
multiple plotting functions on the same figure, such as plotting a morphology and the
voxel cloud of its axon.
