import math, numpy as np
from .morphologies import Compartment


def depth_first_branches(adjacency_list, node=0, return_visited=False):
    branches = []
    visited = set([])

    def dfs_branch(branch, node):
        # Grow the branch with current node.
        branch.append(node)
        # Mark this node as visited
        visited.add(node)
        # Get the child nodes and filter out visited nodes (for networks with loops)
        next_nodes = list(adjacency_list[node] - visited)
        # Continue/create branches if this node has any unvisited children.
        if len(next_nodes) > 0:
            # Continue the branch with the first child.
            branch = dfs_branch(branch, next_nodes[0])
            # Create a new branch for each other child node.
            for new_branch_next_node in next_nodes[1:]:
                # Start a branch from this node, continue it from the child node and pass it
                branches.append(dfs_branch([node], new_branch_next_node))
        return branch

    # Start the first branch at the root node.
    branches.append(dfs_branch([], node))
    if return_visited:
        return branches, visited
    else:
        return branches


def all_depth_first_branches(adjacency_list):
    all_branches = []
    all_nodes = list(range(len(adjacency_list)))
    visited = set()
    first_unvisited_node = 0
    while set(all_nodes) - visited:
        new_branches, new_visits = depth_first_branches(
            adjacency_list, node=first_unvisited_node, return_visited=True
        )
        all_branches.extend(new_branches)
        visited.update(new_visits)
        for node in all_nodes:
            if not node in visited:
                first_unvisited_node = node
                break
    return all_branches


def get_branch_points(branch_list):
    return list(map(lambda b: b[0], branch_list))


def reduce_branch(branch, branch_points):
    reduced_branch = [branch[0]]
    for bp in branch:
        if bp in branch_points:
            reduced_branch.append(bp)
    reduced_branch.append(branch[-1])
    return reduced_branch


class Branch:
    def __init__(self, compartments, parent=None, ordered=True):
        self.origin = compartments[0].start
        self._compartments = compartments
        self._root = compartments[0]
        self.is_root = parent is None
        self._parent_branch = parent
        self.child_branches = set()
        # Are the compartments provided in the order they are connected in?
        if ordered:
            # Iterate over the compartments to set the previous/next as parent/child.
            for c in range(len(compartments) - 1):
                compartments[c]._child = compartments[c + 1]
            for c in range(1, len(compartments)):
                compartments[c]._parent = compartments[c - 1]
            # Polish the start and end of the branch
            compartments[0]._parent = None
            compartments[-1]._child = None
        else:
            raise NotImplementedError(
                "Branches can only be initialized by an ordered array of compartments."
            )

    def add_branch(self, branch):
        self.child_branches.add(branch)

    def __iter__(self):
        return iter(self.walk())

    def walk(self, start=None):
        if start is None:
            start = self._root
        while start._child is not None:
            yield start
            start = start._child

    def interpolate(self, resolution):
        for comp in self._compartments:
            length_comp = np.linalg.norm(comp.end - comp.start)
            if length_comp > resolution:
                num_to_add = math.ceil(length_comp / self.resolution)
                added_points = self.split(comp, num_to_add)

    def split(self, compartment, n):
        """
            Split the compartment in n pieces and make those a part of the branch.

            This function stores a link to the original compartment in the partial
            compartments in the attribute `_original`.

            :param compartment: The compartment to split.
            :type compartment: :class:`.morphologies.Compartment`
            :param n: The amount of pieces to split the compartment into.
            :type n: int
        """
        if n < 2:
            raise ValueError("Compartments must be split in at least 2 parts.")
        new_compartments = []
        # For each dimension calculate n breakpoints between the start and end of the
        # compartment. The arrays for each dimensions are then stacked as columns of a
        # matrix with all the breakpoints. This matrix includes the start and end points
        # themselves.
        points = np.column_stack(
            [
                np.linspace(start, end, n + 1)
                for start, end in zip(compartment.start, compartment.end)
            ]
        )
        # Inside of the loop `last_compartment` is used as a parent for the new
        # compartments. By providing the original compartment's parent as an initial
        # condition, we will connect the first new compartment to the original parent,
        # thus continuing the branch from the first new compartment.
        last_compartment = compartment._parent
        # Loop over the points and use every point except the last as the starting point
        # for a new compartment. Then connect each new compartment to the previous
        # compartment.
        for i in range(len(points) - 1):
            # Copy the compartment information but change start and end.
            c = Compartment.from_template(compartment, start=points[i], end=points[i + 1])
            new_compartments.append(c)
            # Store a reference to the original compartment for back referencing.
            c._original = compartment
            # Connect the new child to its parent
            c._parent = last_compartment
            if last_compartment:
                # Connect to the parent to its child
                last_compartment._child = c
            # Move up the pointer of the loop so that the next compartment is connected to
            # the current one.
            last_compartment = c
        # Finally we should connect the last compartment to the child of the original
        # compartment to continue the branch
        last_compartment._child = compartment._child
        # Remove the original compartment as a part of the branch
        self._compartments.remove(compartment)
        # Add the new compartments into the internal array of compartments
        self._compartments.extend(new_compartments)
        # Is the original compartment was the root of the branch?
        if compartment is self._root:
            # Then we need to replace the root with the first new compartment so that
            # branch iteration starts from the new compartment.
            self._root = new_compartments[0]


class FiberMorphology:
    def __init__(self, compartments):
        self.root_branches = [Branch(compartments)]
