import abc
import itertools
import typing
from collections import deque
from functools import reduce

import morphio
import numpy as np

from ... import config
from ..._encoding import EncodedLabels
from ...config import types
from .. import Morphology

if typing.TYPE_CHECKING:
    from ...storage._files import FileDependency


@config.dynamic(attr_name="parser", auto_classmap=True, default="bsb")
class MorphologyParser:
    cls: type = config.attr(type=types.class_(), default="bsb.morphologies.Morphology")
    branch_cls: type = config.attr(type=types.class_(), default="bsb.morphologies.Branch")

    @abc.abstractmethod
    def parse(self, file: typing.Union["FileDependency", str]) -> Morphology:
        """Parse the morphology"""
        pass


@config.node
class BsbParser(MorphologyParser, classmap_entry="bsb"):
    tags: dict[typing.Union[str, list[str]]] = config.attr(
        type=types.dict(type=types.or_(types.str(), types.list(str)))
    )
    """
    Dictionary mapping SWC tags to sets of morphology labels.
    """
    skip_boundary_labels: list[str] = config.attr(type=types.list(str))
    """
    A set of labels that is used to create gaps in a morphology at certain boundaries.
    No point will be inferred between a child branch of a branch labelled with the given
    labels; usually used to skip points between the soma and its child branches.
    """

    def parse(self, file: typing.Union["FileDependency", str]):
        from ...storage._files import FileDependency

        if not isinstance(file, FileDependency):
            file = FileDependency(file)
        content, encoding = file.get_content(check_store=False)
        return self.parse_content(content.decode(encoding or "utf8"))

    def parse_content(self, content: str):
        data = self._swc_parse(content)
        return self._swc_data_to_morpho(data)

    def _swc_parse(self, content: str):
        try:
            data = [
                swc_data
                for line in content.split("\n")
                if not line.strip().startswith("#")
                and (swc_data := [float(x) for x in line.split() if x != ""])
            ]
        except Exception:
            raise RuntimeError(f"Could not parse SWC content")
        err_lines = ", ".join(str(i) for i, d in enumerate(data) if len(d) != 7)
        if err_lines:
            raise ValueError(f"SWC incorrect on lines: {err_lines}")
        return np.array(data)

    def _swc_data_to_morpho(self, data):
        data = np.array(data, copy=False)
        tag_map = {1: "soma", 2: "axon", 3: "dendrites"}
        if self.tags is not None:
            tag_map.update((int(k), v) for (k, v) in self.tags.items())
        # `data` is the raw SWC data, `samples` and `parents` are the graph nodes and edges.
        samples = data[:, 0].astype(int)
        # Map possibly irregular sample IDs (SWC spec allows this) to an ordered 0 to N map.
        id_map = dict(zip(samples, itertools.count()))
        id_map[-1] = -1
        # Create an adjacency list of the graph described in the SWC data
        adjacency = {n: [] for n in range(len(samples))}
        adjacency[-1] = []
        map_ids = np.vectorize(id_map.get)
        parents = map_ids(data[:, 6])
        for s, p in enumerate(parents):
            adjacency[p].append(s)
        # Now turn the adjacency list into a list of unbranching stretches of the graph.
        # Call these `node_branches` because they only contain the sample/node ids.
        node_branches = []
        for root_node in adjacency[-1]:
            self._swc_branch_dfs(adjacency, node_branches, root_node, data, tag_map)
        branches = []
        roots = []
        _len = sum(len(s[1]) for s in node_branches)
        points = np.empty((_len, 3))
        radii = np.empty(_len)
        tags = np.empty(_len, dtype=int)
        labels = EncodedLabels.none(_len)
        # Now turn each "node branch" into an actual branch by looking up the node data in the
        # samples array. We copy over the node data into several contiguous matrices that will
        # form the basis of the Morphology data structure.
        ptr = 0
        for parent, branch_nodes in node_branches:
            node_data = data[branch_nodes]
            nptr = ptr + len(node_data)
            # Example with the points data matrix: copy over the swc data into contiguous arr
            points[ptr:nptr] = node_data[:, 2:5]
            # Then create a partial view into that data matrix for the branch
            branch_points = points[ptr:nptr]
            # Same here for radius,
            radii[ptr:nptr] = node_data[:, 5]
            branch_radii = radii[ptr:nptr]
            # the SWC tags
            tags[ptr:nptr] = node_data[:, 1]
            if len(branch_nodes) > 1:
                # Since we add an extra point we have to copy its tag from the next point.
                tags[ptr] = tags[ptr + 1]
            branch_tags = tags[ptr:nptr]
            # And the labels
            branch_labels = labels[ptr:nptr]
            for v in np.unique(branch_tags):
                u_tags = tag_map.get(v, f"tag_{v}")
                branch_labels.label(
                    [u_tags] if isinstance(u_tags, str) else u_tags, branch_tags == v
                )
            ptr = nptr
            # Use the views to construct the branch
            branch = self.branch_cls(branch_points, branch_radii, branch_labels)
            branch.set_properties(tags=branch_tags)
            branches.append(branch)
            if parent is not None:
                branches[parent].attach_child(branch)
            else:
                roots.append(branch)
        # Then save the shared data matrices on the morphology
        morpho = self.cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}))
        # And assert that this shared buffer mode succeeded
        assert morpho._check_shared(), "SWC import didn't result in shareable buffers."
        return morpho

    def _swc_branch_dfs(self, adjacency, branches, node, data, tags):
        boundaries = set()
        if self.skip_boundary_labels:
            tset = set(self.skip_boundary_labels)
            for tag, labels in tags.items():
                lset = set(labels if not isinstance(labels, str) else [labels])
                if tset.issuperset(lset):
                    boundaries.add(tag)
        branch = []
        branch_id = len(branches)
        branches.append((None, branch))
        node_stack = deque()
        while True:
            if node is not None:
                # Append the current node to the current branch, and get its child nodes
                branch.append(node)
                child_nodes = adjacency[node]

            if not child_nodes:
                # No children, pop next branch
                try:
                    parent_bid, parent, node, skip = node_stack.pop()
                except IndexError:
                    # No next branch, we're done
                    break
                else:
                    # Start the next branch
                    branch = [] if skip else [parent]
                    branch_id = len(branches)
                    branches.append((parent_bid, branch))
            elif len(child_nodes) == 1 and not (
                data[node, 1] in boundaries and data[child_nodes[0], 1] not in boundaries
            ):
                # One child, and not a skipped boundary: grow the branch
                node = child_nodes[0]
            else:
                # Branch point: create 1 new branch per child point
                # If skip is False we add the current node to all the child branches.
                skip = data[node, 1] in boundaries
                node_stack.extend(
                    (branch_id, node, child, skip) for child in reversed(child_nodes)
                )
                child_nodes = []
                node = None


# Wrapper to append our own attributes to morphio somas and treat it like any other branch
class _MorphIoSomaWrapper:
    def __init__(self, obj):
        self._o = obj

    def __getattr__(self, attr):
        return getattr(self._o, attr)


@config.node
class MorphIOParser(MorphologyParser, classmap_entry="morphio"):
    @config.property(type=types.list(type=types.in_(morphio.Option.__members__)))
    def flags(self):
        return getattr(self, "_flags", morphio.Option.no_modifier)

    @flags.setter
    def flags(self, values):
        self._flags = reduce(
            morphio.Option.__or__,
            [getattr(morphio.Option, flag) for flag in values or []],
            morphio.Option.no_modifier,
        )

    def parse(self, file: typing.Union["FileDependency", str]) -> Morphology:
        from ...storage._files import FileDependency

        if isinstance(file, str):
            file = FileDependency(file)

        with file.provide_locally() as (fp, encoding):
            morpho_io = morphio.Morphology(fp, self.flags)
        # We create shared buffers for the entire morphology, which optimize operations on the
        # entire morphology such as `.flatten`, subtree transformations and IO.  The branches
        # have views on those buffers, and as long as no points are added or removed, we can
        # keep working in shared buffer mode.
        soma = _MorphIoSomaWrapper(morpho_io.soma)
        _len = len(morpho_io.points) + len(soma.points)
        points = np.empty((_len, 3))
        radii = np.empty(_len)
        tags = np.empty(_len, dtype=int)
        labels = EncodedLabels.none(_len)
        soma.children = morpho_io.root_sections
        section_stack = deque([(None, soma)])
        branch = None
        roots = []
        ptr = 0
        while True:
            try:
                parent, section = section_stack.pop()
            except IndexError:
                break
            else:
                nptr = ptr + len(section.points)
                # Fill the branch data into the shared buffers and create views into them.
                points[ptr:nptr] = section.points
                branch_points = points[ptr:nptr]
                radii[ptr:nptr] = section.diameters / 2
                branch_radii = radii[ptr:nptr]
                tags[ptr:nptr] = np.ones(len(section.points), dtype=int) * int(
                    section.type
                )
                branch_tags = tags[ptr:nptr]
                branch_labels = labels[ptr:nptr]
                ptr = nptr
                # Pass the shared buffer views to the branch
                branch = self.branch_cls(branch_points, branch_radii, branch_labels)
                branch.set_properties(tags=branch_tags)
                if parent:
                    parent.attach_child(branch)
                else:
                    roots.append(branch)
                children = reversed([(branch, child) for child in section.children])
                section_stack.extend(children)
        morpho = self.cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}))
        assert (
            morpho._check_shared()
        ), "MorphIO import didn't result in shareable buffers."
        return morpho


__all__ = ["BsbParser", "MorphIOParser", "MorphologyParser"]
