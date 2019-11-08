#############################
List of connection strategies
#############################

Connection strategies starting with ":class:`Connectome" are made for specific cell <.connectivity.Connectome" are made for specific cell>`
connection, those without can be used as general strategies.

* ``from_cell_types``: An array of objects with a ``type`` key indicating presynaptic
  cell types and optionally a ``compartments`` key for an array of compartment types::

    "from_cell_types": [
      {"type": "basket_cell", "compartments": ["axon"]},
      {"type": "stellate_cell", "compartments": ["axon"]}
    ]

* ``to_cell_types``: Same as ``from_cell_types`` but for the postsynaptic cell type.

:class:`TouchingConvergenceDivergence <.connectivity.TouchingConvergenceDivergence>`
====================================================================================

* ``divergence``: Preferred amount of connections starting from 1 from_cell
* ``convergence``: Preferred amount of connections ending on 1 to_cell

:class:`ConnectomeGlomerulusGranule <.connectivity.ConnectomeGlomerulusGranule>`
================================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the dendrite length configured in the granule cell morphology.

:class:`ConnectomeGlomerulusGolgi <.connectivity.ConnectomeGlomerulusGolgi>`
============================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the dendrite radius configured in the Golgi cell morphology.

:class:`ConnectomeGolgiGlomerulus <.connectivity.ConnectomeGolgiGlomerulus>`
============================================================================

Inherits from TouchingConvergenceDivergence. No additional configuration.
Uses the ``axon_x``, ``axon_y``, ``axon_z`` from the Golgi cell morphology
to intersect a parallelopipid Golgi axonal region with the glomeruli.

:class:`ConnectomeGranuleGolgi <.connectivity.ConnectomeGranuleGolgi>`
======================================================================

Creates 2 connectivity sets by default *ascending_axon_to_golgi* and
*parallel_fiber_to_golgi* but these can be overwritten by providing ``tag_aa``
and/or ``tag_pf`` respectively.

Calculates the distance in the XZ plane between granule cells and Golgi cells and
uses the Golgi cell morphology's dendrite radius to decide on the intersection.

Also creates an ascending axon height for each granule cell.

* ``aa_convergence``: Preferred amount of ascending axon synapses on 1 Golgi cell.
* ``pf_convergence``: Preferred amount of parallel fiber synapses on 1 Golgi cell.

:class:`ConnectomeGolgiGranule <.connectivity.ConnectomeGolgiGranule>`
======================================================================

No configuration, it connects each Golgi to each granule cell that it shares a
connected glomerules with.

:class:`ConnectomeAscAxonPurkinje <.connectivity.ConnectomeAscAxonPurkinje>`
============================================================================

Intersects the rectangular extension of the Purkinje dendritic tree with the granule
cells in the XZ plane, uses the Purkinje cell's placement attributes ``extension_x``
and ``extension_z``.

* ``extension_x``: Extension of the dendritic tree in the X plane
* ``extension_z``: Extension of the dendritic tree in the Z plane

:class:`ConnectomePFPurkinje <.connectivity.ConnectomePFPurkinje>`
==================================================================

No configuration. Uses the Purkinje cell's placement attribute ``extension_x``.
Intersects Purkinje cell dendritic tree extension along the x axis with the x position
of the granule cells, as the length of a parallel fiber far exceeds the simulation
volume.
