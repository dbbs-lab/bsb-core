########
Scaffold
########

:class:`~.core.Scaffold` is the main object of the BSB infrastructure (see the
:doc:`/getting-started/top-level-guide` for an introduction to this class).

Properties
----------

The Scaffold object tights together the network description
of the ``Configuration`` with the data stored in the :class:`~.storage.Storage`.
You can access the latter classes with respectively the
:meth:`scaffold.configuration <.core.Scaffold.configuration>` and the
:meth:`scaffold.storage <.core.Scaffold.storage>` attributes.
Scaffold also provides a direct access to all of its main configuration components as class attributes:

- :attr:`scaffold.network <.core.Scaffold.network>` -> :class:`~.config._config.NetworkNode`
- :attr:`scaffold.regions <.core.Scaffold.regions>` -> :class:`~.topology.region.Region`
- :attr:`scaffold.partitions <.core.Scaffold.partitions>` -> :class:`~.topology.partition.Partition`
- :attr:`scaffold.cell_types <.core.Scaffold.cell_types>` -> :class:`~.cell_types.CellType`
- :attr:`scaffold.morphologies <.core.Scaffold.morphologies>` -> :class:`~.morphologies.Morphology`
- :attr:`scaffold.morphologies <.core.Scaffold.morphologies>` -> :class:`~.cell_types.CellType`
- :attr:`scaffold.placement <.core.Scaffold.placement>` -> :class:`~.placement.strategy.PlacementStrategy`
- :attr:`scaffold.connectivity <.core.Scaffold.connectivity>` -> :class:`~.connectivity.strategy.ConnectionStrategy`
- :attr:`scaffold.simulations <.core.Scaffold.simulations>` -> :class:`~.simulation.simulation.Simulation`
- :attr:`scaffold.after_placement <.core.Scaffold.after_placement>` -> :class:`~.postprocessing.AfterPlacementHook`
- :attr:`scaffold.after_connectivity <.core.Scaffold.after_connectivity>` -> :class:`~.postprocessing.AfterConnectivityHook`

There are also a list of methods starting with ``get_`` that allows you to retrieve these components with some
additional filtering parameters (:meth:`get_cell_types <.core.Scaffold.get_cell_types>`,
:meth:`get_placement <.core.Scaffold.get_placement>`,
:meth:`get_placement_of <.core.Scaffold.get_placement_of>`,
:meth:`get_connectivity <.core.Scaffold.get_connectivity>`)

Workflow methods
----------------

Scaffold contains also all the functions required to run the reconstruction pipeline, and to simulate
the resulting networks.
You can run the full reconstruction with the :meth:`compile <.core.Scaffold.compile>` method or any of its sub-step:

- Topology creation / update: :meth:`resize <.core.Scaffold.resize>`
- Cell placement: :meth:`run_placement <.core.Scaffold.run_placement>`
- After placement hook: :meth:`run_after_placement <.core.Scaffold.run_after_placement>`
- Cell connectivity: :meth:`run_connectivity <.core.Scaffold.run_connectivity>`
- After placement hook: :meth:`run_after_connectivity <.core.Scaffold.run_after_connectivity>`
- Run a simulation: :meth:`run_simulation <.core.Scaffold.run_simulation>`

Similarly, you can clear the results of the reconstruction stored so far with the :meth:`clear <.core.Scaffold.clear>`
or any of its sub-step:

- Cell placement: :meth:`clear_placement <.core.Scaffold.clear_placement>`
- Cell connectivity: :meth:`clear_connectivity <.core.Scaffold.clear_connectivity>`

Get Stored data
---------------

You can also inspect the data produced during the reconstruction from the storage:

- :class:`~.storage.interfaces.PlacementSet` from :meth:`get_placement_set <.core.Scaffold.get_placement_set>`,
  :meth:`get_placement_sets <.core.Scaffold.get_placement_sets>`
- :class:`~.storage.interfaces.ConnectivitySet` from :meth:`get_connectivity_set <.core.Scaffold.get_connectivity_set>`,
  :meth:`get_connectivity_sets <.core.Scaffold.get_connectivity_sets>`
