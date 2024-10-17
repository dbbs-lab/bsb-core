########
Scaffold
########

Main object of the BSB infrastructure.

Using the scaffold object, you can inspect the data in the storage by using the
:class:`~.storage.interfaces.PlacementSet` and
:class:`~.storage.interfaces.ConnectivitySet` APIs. PlacementSets can be obtained with
:meth:`scaffold.get_placement_set <.core.Scaffold.get_placement_set>`, and
ConnectivitySets with :meth:`scaffold.get_connectivity_set
<.core.Scaffold.get_placement_set>`.