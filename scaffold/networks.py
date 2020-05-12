import math, numpy as np, random
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
    """
        Reduce a branch (list of points) to only its start and end point and the
        intersection with a list of known branch points.
    """
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
        self._root = _get_root_compartment(compartments[0])
        self._terminus = _get_terminal_compartment(compartments[-1])
        self.is_root = parent is None
        self._parent_branch = parent
        self.child_branches = []
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

    def add_branch(self, branch):
        branch._parent_branch = self
        self.child_branches.append(branch)

    def __iter__(self):
        return iter(self.walk())

    def walk(self, start=None):
        if start is None:
            start = self._root
        while start._child is not None:
            yield start
            start = start._child

    def append(self, compartment):
        self._compartments.append(compartment)
        self._terminus._child = compartment
        compartment._parent = self._terminus
        self._terminus = compartment

    def interpolate(self, resolution):
        for comp in self._compartments:
            length_comp = np.linalg.norm(comp.end - comp.start)
            if length_comp > resolution:
                num_to_add = math.ceil(length_comp / resolution)
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

    def voxelize(self, position, bounding_box, voxel_tree, map):
        for v, comp in enumerate(self._compartments):
            # Check if the current bounding_box needs to be extended
            bounding_box[0] = np.minimum(bounding_box[0], comp.end + position)
            bounding_box[1] = np.maximum(bounding_box[1], comp.end + position)

            # Find the external points of the voxel surrounding the compartment in absolute coordinates
            voxel_bottom_left = np.minimum(comp.start, comp.end)
            voxel_top_right = np.maximum(comp.start, comp.end)
            # Add the voxel to the tree
            voxel_tree.insert(
                v,
                tuple(
                    np.concatenate(
                        (voxel_bottom_left + position, voxel_top_right + position)
                    )
                ),
            )
            map.append(comp)
        return bounding_box, voxel_tree, map


def _init_child_compartments(compartments):
    # Reset/init child compartments
    for c in compartments:
        c._children = []
    # Add child nodes to their parent's children
    for node in compartments:
        if node.parent not in compartments:
            node.parent = None
        if node.parent is not None:
            node.parent._children.append(node)


def _get_root_compartment(compartment):
    while compartment.parent is not None:
        compartment = compartment.parent
    return compartment


def _get_terminal_compartment(compartment):
    while len(compartment._children) == 1:
        compartment = compartment._children[0]
    return compartment


def _consume_branch(unvisited, root_compartment, parent=None):
    branch = Branch([root_compartment], parent=parent, ordered=False)
    unvisited.remove(root_compartment)
    root_compartment._parent = None
    compartment = root_compartment
    while len(compartment._children) == 1:
        next_compartment = compartment._children[0]
        branch.append(next_compartment)
        unvisited.remove(next_compartment)
        compartment = next_compartment
    if len(compartment._children) > 0:
        for child in compartment._children:
            child_branch = _consume_branch(unvisited, child, parent=branch)
            branch.add_branch(child_branch)
    return branch


def _copy_linked_compartments(compartments):
    copy_map = {}
    new_compartments = []
    for c in compartments:
        new_c = Compartment.from_template(c)
        copy_map[new_c.id] = new_c
        new_compartments.append(new_c)
    for c in new_compartments:
        if c.parent_id in copy_map:
            c.parent = copy_map[c.parent_id]
        else:
            c.parent_id = -1
            c.parent = None
    return new_compartments


def create_root_branched_network(compartments):
    root_branches = []
    _init_child_compartments(compartments)
    unvisited = set(compartments)
    while len(unvisited) > 0:
        starting_compartment = next(iter(unvisited))
        root_compartment = _get_root_compartment(starting_compartment)
        root_branch = _consume_branch(unvisited, root_compartment)
        root_branches.append(root_branch)
    return root_branches


class FiberMorphology:
    def __init__(self, compartments):
        compartments = _copy_linked_compartments(compartments)
        self.root_branches = create_root_branched_network(compartments)
