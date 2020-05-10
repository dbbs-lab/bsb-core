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


def split_branches(branch_list):
    branch_points = get_branch_points(branch_list)
    split_branches = []
    for branch in branch_list:
        pass

    return split_branches


class Branch:
    def __init__(self, compartments, is_root=False):
        self.origin = compartments[0].start
        self._compartments = compartments
        self._root = compartments[0]
        # Spoof branch connectivity by assuming it is ordered as they are connected
        # TODO: Use compartment.parent to assign _child. (Check correct conn? No -> slow)
        for c in range(len(compartments) - 1):
            compartments[c]._child = compartments[c + 1]
        self.child_branches = set()
        self.is_root = is_root

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


class FiberMorphology:
    def __init__(self, compartments):
        self.root_branches = [Branch(compartments, is_root=True)]
