# TODO: to be cleaned up and useful parts integrated into morphologies.


class Branch2:
    def interpolate(self, resolution):
        for comp in self._compartments:
            length_comp = np.linalg.norm(comp.end - comp.start)
            if length_comp > resolution + 1e-3:
                num_to_add = math.ceil(length_comp / resolution)
                added_points = self.split(comp, num_to_add)

    def split(self, compartment, n):
        """
        Split the compartment in n pieces and make those a part of the branch.

        This function stores a link to the original compartment in the partial
        compartments in the attribute `_original`.

        :param compartment: The compartment to split.
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
    def __init__(self, compartments, rotation):
        compartments = _copy_linked_compartments(compartments)
        if rotation is None:
            orientation = None
        else:
            orientation = np.array(
                [np.cos(rotation[0]), np.sin(rotation[0]), np.sin(rotation[1])]
            )
        self.root_branches = create_root_branched_network(compartments, orientation)

    def flatten(self, branches=None):
        if branches is None:
            branches = self.root_branches
        compartments = []
        for branch in branches:
            compartments.extend(list(branch._compartments))
            compartments.extend(self.flatten(branch.child_branches))
        return compartments
