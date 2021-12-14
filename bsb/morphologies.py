import abc, numpy as np, pickle, h5py, math, itertools
from .helpers import ConfigurableClass
from .voxels import VoxelCloud, detect_box_compartments, Box
from sklearn.neighbors import KDTree
from .exceptions import *
from .reporting import report


class Compartment:
    """
    Compartments are line segments with a radius. They can be constructed from the points
    on a :class:`Branch <.morphologies.Branch>` or by concatenating the results of a
    depth-first iteration of the branches of a :class:`Morphology
    <.morphologies.Morphology>`.
    """

    def __init__(
        self,
        start,
        end,
        radius,
        id=None,
        labels=None,
        parent=None,
        section_id=None,
        morphology=None,
    ):
        self.id = id
        self.start = start
        self.end = end
        self.radius = radius
        self.labels = labels if labels is not None else []
        self.parent = parent
        self.section_id = section_id
        self.morphology = morphology

    @property
    def midpoint(self):
        # Calculate midpoint of the compartment
        if not hasattr(self, "_midpoint"):
            self._midpoint = (self.end - self.start) / 2 + self.start
        return self._midpoint

    @property
    def spherical(self):
        # Calculate the radius of the outer sphere of this compartment
        if not hasattr(self, "_spherical"):
            self._spherical = np.sqrt(np.sum((self.start - self.end) ** 2)) / 2
        return self._spherical

    @classmethod
    def from_template(cls, template, **kwargs):
        """
        Create a compartment based on  a template compartment. Accepts any keyword
        argument to overwrite or add attributes.
        """
        c = cls(
            id=template.id,
            start=template.start,
            end=template.end,
            radius=template.radius,
            labels=template.labels.copy(),
            parent=template.parent,
            section_id=template.section_id,
            morphology=template.morphology,
        )
        for k, v in kwargs.items():
            c.__dict__[k] = v
        return c


class NilCompartment(Compartment):
    def __init__(self):
        self.id = -1
        self.start = float("nan")
        self.end = float("nan")
        self.radius = float("nan")
        self.labels = []
        self.parent = None
        self.section_id = None
        self.morphology = None


def branch_iter(branch):
    """
    Iterate over a branch and all of its children depth first.
    """
    yield branch
    for child in branch._children:
        yield from branch_iter(child)


def _validate_branch_args(args):
    vec = Branch.vectors
    if len(args) > len(vec):
        raise TypeError(
            f"__init__ takes {len(vec) + 1} arguments but {len(args) + 1} given."
        )
    if len(args) < len(vec):
        n = len(vec) - len(args)
        w = "argument" if n == 1 else "arguments"
        missing = ", ".join(map("'{}'".format, vec[-n:]))
        raise TypeError(f"__init__ is missing {n} required {w}: {missing}.")


class Branch:
    """
    A vector based representation of a series of point in space. Can be a root or
    connected to a parent branch. Can be a terminal branch or have multiple children.
    """

    vectors = ["x", "y", "z", "radii"]

    def __init__(self, *args, labels=None):
        _validate_branch_args(args)
        self._children = []
        self._full_labels = []
        self._label_masks = {}
        self._parent = None
        for v, vector in enumerate(self.__class__.vectors):
            self.__dict__[vector] = args[v]

    @property
    def points(self):
        """
        Return the vectors of this branch as a matrix.
        """
        return np.column_stack(tuple(getattr(self, v) for v in self.__class__.vectors))

    @property
    def size(self):
        """
        Returns the amount of points on this branch

        :returns: Number of points on the branch.
        :rtype: int
        """
        return len(getattr(self, self.__class__.vectors[0]))

    @property
    def terminal(self):
        """
        Returns whether this branch is terminal or has children.

        :returns: True if this branch has no children, False otherwise.
        :rtype: bool
        """
        return not self._children

    def label(self, *labels):
        """
        Add labels to every point on the branch. See :func:`label_points
        <.morphologies.Morphology.label_points>` to label individual points.

        :param labels: Label(s) for the branch.
        :type labels: str
        """
        self._full_labels.extend(labels)

    def label_points(self, label, mask):
        """
        Add labels to specific points on the branch. See :func:`label
        <.morphologies.Morphology.label>` to label the entire branch.

        :param label: Label to apply to the points.
        :type label: str
        :param mask: Boolean mask equal in size to the branch that determines which points get labelled.
        :type mask: np.ndarray(dtype=bool, shape=(branch_size,))
        """
        self._label_masks[label] = np.array(mask, dtype=bool)

    @property
    def children(self):
        """
        Collection of the child branches of this branch.

        :returns: list of :class:`Branches <.morphologies.Branch>`
        :rtype: list
        """
        return self._children.copy()

    def attach_child(self, branch):
        """
        Attach a branch as a child to this branch.

        :param branch: Child branch
        :type branch: :class:`Branch <.morphologies.Branch>`
        """
        if branch._parent is not None:
            branch._parent.detach_child(branch)
        self._children.append(branch)
        branch._parent = self

    def detach_child(self, branch):
        """
        Remove a branch as a child from this branch.

        :param branch: Child branch
        :type branch: :class:`Branch <.morphologies.Branch>`
        """
        try:
            self._children.remove(branch)
            branch._parent = None
        except ValueError:
            raise ValueError("Branch could not be detached, it is not a child branch.")

    def to_compartments(self, start_id=0, parent=None):
        """
        Convert the branch to compartments.

        .. deprecated:: 3.6
            Use the vectors and points API instead (``.points``, ``.walk()``)
        """
        comp_id = start_id

        def to_comp(data, labels):
            nonlocal comp_id, parent
            kwargs = dict(id=comp_id, parent=parent, labels=labels)
            if hasattr(self, "_neuron_sid"):
                kwargs["section_id"] = self._neuron_sid
            comp = Compartment(*data, **kwargs)
            comp_id += 1
            parent = comp
            return comp

        # Walk over each pair of points as the start and end if a compartment.
        # Start from the end of the parent branch's last compartment.
        comps = [
            to_comp(data, labels)
            for data, labels in _pairwise_iter(self.walk(), self.label_walk())
        ]
        return comps

    def walk(self):
        """
        Iterate over the points in the branch.
        """
        return zip(*(self.__dict__[v] for v in self.__class__.vectors))

    def label_walk(self):
        """
        Iterate over the labels of each point in the branch.
        """
        labels = self._full_labels.copy()
        n = self.size
        shared = np.ones((n, len(labels)), dtype=bool)
        labels.extend(self._label_masks.keys())
        label_row = np.array(labels)
        label_matrix = np.column_stack((shared, *self._label_masks.values()))
        return (label_row[label_matrix[i, :]] for i in range(n))

    def has_label(self, label):
        return label in self._full_labels

    def has_any_label(self, labels):
        return any(self.has_label(l) for l in labels)

    def get_labelled_points(self, label):
        point_label_iter = zip(self.walk(), self.label_walk())
        return list(p for p, labels in point_label_iter if label in labels)


def _pairwise_iter(walk_iter, labels_iter):
    try:
        start = np.array(next(walk_iter)[:3])
        # Throw away the first point's labels as there are only n - 1 compartments.
        _ = next(labels_iter)
    except StopIteration:
        return iter(())
    for data in walk_iter:
        end = np.array(data[:3])
        radius = data[3]
        labels = next(labels_iter)
        yield (start, end, radius), labels
        start = end


class Morphology:
    """
    A multicompartmental spatial representation of a cell based on connected 3D
    compartments.

    :todo: Uncouple from the MorphologyRepository and merge with TrueMorphology.
    """

    def __init__(self, roots):
        self.cloud = None
        self.has_morphology = True
        self.has_voxels = False
        self.roots = roots
        self._compartments = None
        self.update_compartment_tree()

    @property
    def compartments(self):
        if self._compartments is None:
            self._compartments = self.to_compartments()
        return self._compartments

    @property
    def branches(self):
        """
        Return a depth-first flattened array of all branches.
        """
        return self.get_branches()

    def get_branches(self, labels=None):
        """
        Return a depth-first flattened array of all or the selected branches.

        :param labels: Names of the labels to select.
        :type labels: list
        :returns: List of all branches or all branches with any of the labels
          when given
        :rtype: list
        """
        root_iter = (branch_iter(root) for root in self.roots)
        all_branch_iter = itertools.chain(*root_iter)
        if labels is None:
            return list(all_branch_iter)
        else:
            return [b for b in all_branch_iter if b.has_any_label(labels)]

    def to_compartments(self):
        """
        Return a flattened array of compartments
        """
        comp_counter = 0

        def treat_branch(branch, last_parent=None):
            nonlocal comp_counter
            comps = branch.to_compartments(comp_counter, last_parent)
            comp_counter += len(comps)
            # If this branch has no compartments just pass on the compartment we were
            # supposed to connect our first compartment to. That way this empty branch
            # is skipped and the next compartments are still connected in the comp tree.
            parent_comp = comps[-1] if len(comps) else last_parent
            child_iters = (treat_branch(b, parent_comp) for b in branch._children)
            return itertools.chain(comps, *child_iters)

        return [*itertools.chain(*(treat_branch(root) for root in self.roots))]

    def flatten(self, vectors=None, matrix=False, labels=None):
        """
        Return the flattened vectors of the morphology

        :param vectors: List of vectors to return such as ['x', 'y', 'z'] to get the
          positional vectors.
        :type vectors: list of str
        :returns: Tuple of the vectors in the given order, if `matrix` is True a
          matrix composed of the vectors is returned instead.
        :rtype: tuple of ndarrays (`matrix=False`) or matrix (`matrix=True`)
        """
        if vectors is None:
            vectors = Branch.vectors
        branches = self.get_branches(labels=labels)
        if not branches:
            if matrix:
                return np.empty((0, len(vectors)))
            return tuple(np.empty(0) for _ in vectors)
        t = tuple(np.concatenate(tuple(getattr(b, v) for b in branches)) for v in vectors)
        return np.column_stack(t) if matrix else t

    def update_compartment_tree(self):
        # Eh, this code will be refactored soon, if you're still seeing this in v4 open an
        # issue.
        if self.compartments:
            self.compartment_tree = KDTree(np.array([c.end for c in self.compartments]))

    def voxelize(self, N, compartments=None):
        self.cloud = VoxelCloud.create(self, N, compartments=compartments)

    def create_compartment_map(self, tree, boxes, voxels, box_size):
        compartment_map = []
        box_positions = np.column_stack(boxes[:, voxels])
        for i in range(box_positions.shape[0]):
            box_origin = box_positions[i, :]
            compartment_map.append(detect_box_compartments(tree, box_origin, box_size))
        return compartment_map

    def get_bounding_box(self, compartments=None, centered=True):
        # Use the compartment tree to get a quick array of the compartments positions
        compartment_positions = np.array(
            list(map(lambda c: c.midpoint, compartments or self.compartments))
        )
        # Determine the amount of dimensions of the morphology. Let's hope 3 ;)
        n_dimensions = range(compartment_positions.shape[1])
        # Create a bounding box
        outer_box = Box()
        # The outer box dimensions are equal to the maximum distance between compartments in each of n dimensions
        outer_box.dimensions = np.array(
            [
                np.max(compartment_positions[:, i]) - np.min(compartment_positions[:, i])
                for i in n_dimensions
            ]
        )
        # The outer box origin should be in the middle of the outer bounds if 'centered' is True. (So lowermost point + sometimes half of dimensions)
        outer_box.origin = np.array(
            [
                np.min(compartment_positions[:, i])
                + (outer_box.dimensions[i] / 2) * int(centered)
                for i in n_dimensions
            ]
        )
        return outer_box

    def get_search_radius(self, plane="xyz"):
        pos = np.array(self.compartment_tree.get_arrays()[0])
        dimensions = ["x", "y", "z"]
        try:
            max_dists = np.max(
                np.abs(np.array([pos[:, dimensions.index(d)] for d in plane])), axis=1
            )
        except ValueError as e:
            raise ValueError("Unknown dimensions in dimension string '{}'".format(plane))
        return np.sqrt(np.sum(max_dists ** 2))

    def get_compartment_network(self):
        compartments = self.compartments
        node_list = [set([]) for c in compartments]
        # Add child nodes to their parent's adjacency set
        for node in compartments[1:]:
            if node.parent is None:
                continue
            node_list[int(node.parent.id)].add(int(node.id))
        return node_list

    def get_compartment_positions(self, labels=None):
        comp_pos = [c.end for c in self.get_compartments(labels=labels)]
        return np.array(comp_pos).reshape(-1, 3)

    def get_compartment_tree(self, labels=None):
        if labels is None:
            return self.compartment_tree
        else:
            labelled_comps = self.get_compartments(labels=labels)
            if labelled_comps:
                return _compartment_tree(labelled_comps)
            else:
                return None

    def get_compartment_submask(self, labels):
        ## TODO: Remove; voxelintersection & touchdetection audit should make this code
        ## obsolete.
        return [c.id for c in self.get_compartments(labels)]

    def get_compartments(self, labels=None):
        if labels is None:
            return self.compartments.copy()
        return [c for c in self.compartments if any(l in labels for l in c.labels)]

    def rotate(self, v0, v):
        """

        Rotate a morphology to be oriented as vector v, supposing to start from orientation v0.
        norm(v) = norm(v0) = 1
        Rotation matrix R, representing a rotation of angle alpha around vector k

        """
        R = get_rotation_matrix(v0, v)

        for c in range(len(self.compartments)):
            self.compartments[c].start = R.dot(self.compartments[c].start)
            self.compartments[c].end = R.dot(self.compartments[c].end)

        self.update_compartment_tree()


def _compartment_tree(compartments):
    return KDTree(np.array([c.end for c in compartments]))


class EmptyTree:
    pass


class Representation(ConfigurableClass):
    pass


class GranuleCellGeometry(Representation):
    casts = {
        "dendrite_length": float,
        "pf_height": float,
        "pf_height_sd": float,
    }
    required = ["dendrite_length", "pf_height", "pf_height_sd"]

    def validate(self):
        pass


class PurkinjeCellGeometry(Representation):
    def validate(self):
        pass


class GolgiCellGeometry(Representation):
    casts = {
        "dendrite_radius": float,
        "axon_x": float,
        "axon_y": float,
        "axon_z": float,
    }

    required = ["dendrite_radius"]

    def validate(self):
        pass


class RadialGeometry(Representation):
    casts = {
        "dendrite_radius": float,
    }

    required = ["dendrite_radius"]

    def validate(self):
        pass


class NoGeometry(Representation):
    def validate(self):
        pass


def get_rotation_matrix(v0, v):
    I = np.identity(3)
    # Reduce 1-size dimensions
    v0 = np.array(v0).squeeze()
    v = np.array(v).squeeze()
    # Normalize orientation vectors
    v0 = v0 / np.linalg.norm(v0)
    v = v / np.linalg.norm(v0)
    alpha = np.arccos(np.dot(v0, v))

    if math.isclose(alpha, 0.0, rel_tol=1e-4):
        report(
            "Rotating morphology between parallel orientation vectors, {} and {}!".format(
                v0, v
            ),
            level=3,
        )
        # We will not rotate the morphology, thus R = I
        return I
    elif math.isclose(alpha, np.pi, rel_tol=1e-4):
        report(
            "Rotating morphology between antiparallel orientation vectors, {} and {}!".format(
                v0, v
            ),
            level=3,
        )
        # We will rotate the morphology of 180° around a vector orthogonal to
        # the starting vector v0 (the same would be if we take the ending vector
        # v). We set the first and third components to 1; the second one is
        # obtained to have the scalar product with v0 equal to 0
        kx = 1
        kz = 1
        ky = -(v0[0] + v0[2]) / v0[1]
        k = np.array([kx, ky, kz])
    else:
        k = (np.cross(v0, v)) / math.sin(alpha)
        k = k / np.linalg.norm(k)

    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # Compute and return the rotation matrix using Rodrigues' formula
    return I + math.sin(alpha) * K + (1 - math.cos(alpha)) * np.linalg.matrix_power(K, 2)
