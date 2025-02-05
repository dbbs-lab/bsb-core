import abc

import numpy as np

from . import config
from .config import refs
from .exceptions import (
    ConnectivityError,
    DatasetNotFoundError,
    MorphologyDataError,
    MorphologyError,
)
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
class FuseConnections(AfterConnectivityHook):
    """This hook enables the creation of a new connectivity set by chaining the provided connectivity sets.
    For example, if connectivity sets A -> B and B -> C are given, they will be remapped to A -> C..

    :param connections: A list of connectivity names to be merged.
    """

    connections: list[str] = config.list(required=True)

    def postprocess(self):
        class Node:
            def __init__(self, name):
                self.name = name
                self.parents = []
                self.children = []
                self.resolved_cs = [
                    np.empty((0, 3), dtype=int),
                    np.empty((0, 3), dtype=int),
                ]

            def add_child(self, child):
                self.children.append(child)

            def add_parent(self, parent):
                self.parents.append(parent)

            def __eq__(self, name):
                return self.name == name

        # Create the connectivity tree
        tree = []
        roots = []  # store the list of potential root of the tree
        ends = []  # store the list of potential end of the tree
        # convert to set to avoid potential duplicates
        for connection in set(self.connections):
            try:
                cs = self.scaffold.get_connectivity_set(connection)
            except DatasetNotFoundError:
                raise ConnectivityError(
                    f"AfterConnectivityHook {self.name} do not find {connection} ConnectivitySet."
                )
            except ValueError as e:
                raise e
            if cs.pre_type.name not in tree:
                tree.append(Node(name=cs.pre_type.name))
                roots.append(cs.pre_type.name)
            if cs.post_type.name not in tree:
                tree.append(Node(name=cs.post_type.name))
                ends.append(cs.post_type.name)

            tree[tree.index(cs.pre_type.name)].add_child(cs)
            tree[tree.index(cs.post_type.name)].add_parent(cs.pre_type.name)
            if cs.post_type.name in roots:
                roots.remove(cs.post_type.name)
            if cs.pre_type.name in ends:
                ends.remove(cs.pre_type.name)

        if len(roots) != 1 or len(ends) != 1:
            raise ConnectivityError(
                f"Multiple roots or ends detected in your chain of connectivity sets."
            )

        def _return_cs(node, parent_cs):
            if parent_cs is not None and len(node.resolved_cs[0]) > 0:
                return self.merge_sets(parent_cs, node.resolved_cs)
            elif parent_cs is not None:
                return parent_cs
            else:
                return node.resolved_cs

        def visit(node, passed=[], marked=[], parent_cs=None):
            # Depth-first search recursive algorithm to merge connection sets and check for loops
            if node in marked:
                return _return_cs(node, parent_cs)
            if node in passed:
                raise ConnectivityError(
                    "Loop detected in your chain of connectivity sets."
                )
            passed.append(node)
            for out_cs in node.children:
                cs = visit(
                    tree[tree.index(out_cs.post_type.name)],
                    passed,
                    marked,
                    out_cs.load_connections().all(),
                )
                node.resolved_cs = np.concatenate([node.resolved_cs, cs], axis=1)
            marked.append(node)
            return _return_cs(node, parent_cs)

        first_node = tree[tree.index(roots[0])]
        last_node = tree[tree.index(ends[0])]
        new_cs = visit(first_node)

        first_ps = first_node.children[0].pre_type.get_placement_set()
        last_ps = self.scaffold.get_placement_set(last_node.name)
        self.scaffold.connect_cells(first_ps, last_ps, new_cs[0], new_cs[1], self.name)

    def merge_sets(
        self,
        left_set: tuple[np.ndarray, np.ndarray],
        right_set: tuple[np.ndarray, np.ndarray],
    ):

        # sort according to common cell ids
        left_sorting = np.argsort(left_set[1], axis=0)
        left_pre_sorted = left_set[0][left_sorting[:, 0]]
        left_post_sorted = left_set[1][left_sorting[:, 0], 0]
        right_sorting = np.argsort(right_set[0], axis=0)
        right_pre_sorted = right_set[0][right_sorting[:, 0], 0]
        right_post_sorted = right_set[1][right_sorting[:, 0]]

        # get unique common cells and counts
        u1, index1, counts1 = np.unique(
            left_post_sorted, return_index=True, return_counts=True
        )

        u2, index2, counts2 = np.unique(
            right_pre_sorted, return_index=True, return_counts=True
        )

        common1 = np.isin(u1, u2)
        common2 = np.isin(u2, u1)

        # assign the indices to retrieve the positions of all the recurrences of the unique
        left_post_ref = np.array(
            [(index1[i], index1[i + 1]) for i in range(len(index1) - 1)]
        )
        left_post_ref = np.append(
            left_post_ref, [(index1[-1], len(left_post_sorted))], axis=0
        )
        right_pre_ref = np.array(
            [(index2[i], index2[i + 1]) for i in range(len(index2) - 1)]
        )
        right_pre_ref = np.append(
            right_pre_ref, [(index2[-1], len(right_pre_sorted))], axis=0
        )
        # Cycle on the common uniques and combine the pre - post references
        new_size = np.dot(counts2[common2], counts1[common1])
        new_left_pre = np.zeros((new_size, 3), dtype=int)
        new_right_post = np.zeros((new_size, 3), dtype=int)
        cnt = 0
        for l, r in zip(left_post_ref[common1], right_pre_ref[common2]):
            for srs_loc in left_pre_sorted[l[0] : l[1] :]:
                for dest_loc in right_post_sorted[r[0] : r[1] :]:
                    new_left_pre[cnt] = srs_loc
                    new_right_post[cnt] = dest_loc
                    cnt += 1

        return new_left_pre, new_right_post


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
    "FuseConnections",
    "Relay",
    "SpoofDetails",
]
