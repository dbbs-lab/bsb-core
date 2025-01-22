import abc

import numpy as np

from . import config
from .config import refs
from .exceptions import MorphologyDataError, MorphologyError
from .reporting import report


@config.dynamic(attr_name="strategy")
class AfterPlacementHook(abc.ABC):
    name: str = config.attr(key=True)

    def queue(self, pool):
        def static_function(scaffold, name):
            return scaffold.after_placement[name].postprocess()

        pool.queue(static_function, (self.name,), submitter=self)

    @abc.abstractmethod
    def postprocess(self):
        pass


@config.dynamic(attr_name="strategy")
class AfterConnectivityHook(abc.ABC):
    name: str = config.attr(key=True)

    def queue(self, pool):
        def static_function(scaffold, name):
            return scaffold.after_connectivity[name].postprocess()

        pool.queue(static_function, (self.name,), submitter=self)

    @abc.abstractmethod
    def postprocess(self):
        pass


class SpoofDetails(AfterConnectivityHook):
    """
    Create fake morphological intersections between already connected non-detailed
    connection types.
    """

    casts = {"presynaptic": str, "postsynaptic": str}

    def postprocess(self):
        # Check which connection types exist between the pre- and postsynaptic types.
        connection_results = self.scaffold.get_connection_cache_by_cell_type(
            presynaptic=self.presynaptic, postsynaptic=self.postsynaptic
        )
        # Iterate over each involved connectivity matrix
        for connection_result in connection_results:
            connection_type = connection_result[0]
            for connectivity_matrix in connection_result[1:]:
                # Spoof details (morphology & section intersection) between the
                # non-detailed connections in the connectivity matrix.
                self.spoof_connections(connection_type, connectivity_matrix)

    def spoof_connections(self, connection_type, connectivity_matrix):
        from_type = connection_type.presynaptic.type
        to_type = connection_type.postsynaptic.type
        from_entity = False
        to_entity = False
        # Check whether any of the types are relays or entities.
        if from_type.entity:
            from_entity = True
            if to_type.entity:
                raise MorphologyError(
                    "Can't spoof detailed connections between 2 entity cell types."
                )
        elif to_type.entity:
            to_entity = True
        # If they're not relays or entities, load their morphologies
        if not from_entity:
            from_morphologies = from_type.list_all_morphologies()
            if len(from_morphologies) == 0:
                raise MorphologyDataError(
                    "Can't spoof detailed connection without morphologies for "
                    f"'{from_type.name}'"
                )
        if not to_entity:
            to_morphologies = to_type.list_all_morphologies()
            if len(to_morphologies) == 0:
                raise MorphologyDataError(
                    "Can't spoof detailed connection without morphologies for '{}'".format(
                        to_type.name
                    )
                )
        # If they are entities or relays, steal the first morphology of the other cell type.
        # Under no circumstances should entities or relays be represented as actual
        # morphologies, so this should not matter: the data just needs to be spoofed for
        # other parts of the scaffold to function.
        if from_entity:
            from_morphologies = [to_morphologies[0]]
        if to_entity:
            to_morphologies = [from_morphologies[0]]

        # Use only the first morphology for spoofing.
        # At a later point which morphology belongs to which cell should be decided
        # as a property of the cell and not the connection.
        # At that point we can spoof the same morphologies to the opposing relay type.
        #
        # The left column will be the first from_morphology (0) and the right column
        # will be the first to_morphology (1)
        _from = np.zeros(len(connectivity_matrix))
        _to = np.ones(len(connectivity_matrix))
        morphologies = np.column_stack((_from, _to))
        # Generate the map
        morpho_map = [from_morphologies[0], to_morphologies[0]]
        from_m = self.scaffold.morphology_repository.load(from_morphologies[0])
        to_m = self.scaffold.morphology_repository.load(to_morphologies[0])
        # Select random axons and dendrites to connect
        axons = np.array(from_m.get_compartment_submask(["axon"]))
        dendrites = np.array(to_m.get_compartment_submask(["dendrites"]))
        compartments = np.column_stack(
            (
                axons[np.random.randint(0, len(axons), len(connectivity_matrix))],
                dendrites[np.random.randint(0, len(dendrites), len(connectivity_matrix))],
            )
        )
        # Erase previous connection data so that `.connect_cells` can overwrite it.
        self.scaffold.cell_connections_by_tag[connection_type.name] = np.empty((0, 2))
        # Write the new spoofed connection data
        self.scaffold.connect_cells(
            connection_type,
            connectivity_matrix,
            morphologies=morphologies,
            compartments=compartments,
            morpho_map=morpho_map,
        )
        report(
            "Spoofed details of {} connections between {} and {}".format(
                len(connectivity_matrix),
                connection_type.presynaptic.type.name,
                connection_type.postsynaptic.type.name,
            ),
            level=2,
        )


@config.node
class Relay(AfterConnectivityHook):
    """
    Replaces connections on a cell with the relayed connections to the connection targets
    of that cell. Not implemented yet.
    """

    cell_types = config.reflist(refs.cell_type_ref)

    def postprocess(self):
        pass


class BidirectionalContact(AfterConnectivityHook):
    # Replicates all contacts (connections and compartments) to have bidirection in gaps
    def postprocess(self):
        for type in self.types:
            self.scaffold.cell_connections_by_tag[type] = self._invert_append(
                self.scaffold.cell_connections_by_tag[type]
            )
            self.scaffold.connection_compartments[type] = self._invert_append(
                self.scaffold.connection_compartments[type]
            )
            self.scaffold.connection_morphologies[type] = self._invert_append(
                self.scaffold.connection_morphologies[type]
            )

    def _invert_append(self, old):
        return np.concatenate((old, np.stack((old[:, 1], old[:, 0]), axis=1)), axis=0)


__all__ = [
    "BidirectionalContact",
    "AfterPlacementHook",
    "AfterConnectivityHook",
    "Relay",
    "SpoofDetails",
]
