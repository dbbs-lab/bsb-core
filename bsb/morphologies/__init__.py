"""
Sorry robots of the future, this is still just a quick internal stub I haven't properly
finished.

It goes ``morphology-on-file`` into ``repository`` that the ``storage`` needs to provide
support for. Then after a placement job has placed cells for a chunk, the positions are
sent to a ``distributor`` that is supposed to use the ``indicators`` to ask the
``storage.morphology_repository`` which ``loaders`` are appropriate for the given
``selectors``, then, still hopefully using just morpho metadata the  ``distributor``
generates indices and rotations. In more complex cases the ``selector`` and
``distributor`` can both load the morphologies but this will slow things down.

In the simulation step, these (possibly dynamically modified) morphologies are passed
to the cell model instantiators.
"""
import abc, numpy as np, pickle, h5py, math, itertools
from ..voxels import VoxelCloud, detect_box_compartments, Box
from sklearn.neighbors import KDTree
from ..exceptions import *
from ..reporting import report
from .. import config
import operator


class MorphologySet:
    """
    Associates a set of :class:`StoredMorphologies
    <.storage.interfaces.StoredMorphology>` to a dataset of indices and their
    rotations.
    """

    def __init__(self, loaders, m_indices, rotations):
        self._m_indices = m_indices
        self._loaders = loaders
        self._rotations = rotations

    def __len__(self):
        return len(self._m_indices)

    def get_indices(self):
        return self._m_indices

    def get_rotations(self):
        return self._rotations

    def iter_morphologies(self, cache=True):
        if not cache:
            yield from (self._loaders[idx].load() for idx in self._m_indices)
        else:
            _cached = {}
            for idx in zip(self._m_indices, self._rotations):
                if idx not in _cached:
                    _cached[idx] = self._loaders[idx].load()
                yield _cached[idx].copy()

    def _serialize_loaders(self):
        return [loader.get_meta()["name"] for loader in self._loaders]

    def merge(self, other):
        merge_offset = len(self._loaders)
        merged_loaders = self._loaders + other._loaders
        print("merge", self._m_indices, self._m_indices.shape, other._m_indices.shape)
        merged_indices = np.concatenate(
            (self._m_indices, other._m_indices + merge_offset)
        )
        merged_rotations = np.concatenate((self._rotations, other._rotations))
        return MorphologySet(merged_loaders, merged_indices, merged_rotations)


@config.dynamic(
    required=False, default="random", auto_classmap=True, classmap_entry="random"
)
class MorphologyDistributor:
    """
    Distributes morphologies and rotations for a given set of placement indications and
    placed cell positions.

    Config
    ------

    If omitted in the configuration the default ``random`` distributor is used that
    assigns selected morphologies randomly without rotating them.

    .. code-block:: json

      { "placement": { "place_XY": {
        "distributor": {
          "cls": "random"
        }
      }}}
    """

    def distribute(self, cell_type, indicator, positions):
        """
        Uses the morphology selection indicators to select morphologies and
        returns a MorphologySet of randomly assigned morphologies
        """
        selectors = indicator.assert_indication("morphological")
        loaders = self.scaffold.storage.morphologies.select(*selectors)
        if not loaders:
            ids = np.zeros(len(positions))
        else:
            ids = np.random.default_rng().integers(len(loaders), size=len(positions))
        return MorphologySet(loaders, ids, np.zeros((len(positions), 3)))


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
        for v, vector in enumerate(type(self).vectors):
            setattr(self, vector, args[v])

    @property
    def size(self):
        """
        Returns the amount of points on this branch

        :returns: Number of points on the branch.
        :rtype: int
        """
        return len(getattr(self, type(self).vectors[0]))

    def __len__(self):
        return self.size

    @property
    def points(self):
        """
        Return the vectors of this branch as a matrix.
        """
        return self.as_matrix(with_radius=True)

    @property
    def terminal(self):
        """
        Returns whether this branch is terminal or has children.

        :returns: True if this branch has no children, False otherwise.
        :rtype: bool
        """
        return not self._children

    def label_all(self, *labels):
        """
        Add labels to every point on the branch. See :func:`label_points
        <.morphologies.Morphology.label_points>` to label individual points.

        :param labels: Label(s) for the branch.
        :type labels: str
        """
        self._full_labels.extend(labels)

    def label_points(self, label, mask, join=operator.or_):
        """
        Add labels to specific points on the branch. See :func:`label
        <.morphologies.Morphology.label>` to label the entire branch.

        :param label: Label to apply to the points.
        :type label: str
        :param mask: Boolean mask equal in size to the branch that determines which points get labelled.
        :type mask: np.ndarray(dtype=bool, shape=(branch_size,))
        :param join: The operation to use to combine the new labels with the existing
          labels. Defaults to ``|`` (``operator.or_``).
        :type join: operator function
        """
        mask = np.array(mask, dtype=bool)
        if label in self._label_masks:
            labels = self._label_masks[label]
            self._label_masks[label] = join(labels, mask)
        else:
            self._label_masks[label] = mask

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

    def walk(self):
        """
        Iterate over the points in the branch.
        """
        return zip(*(vars(self)[v] for v in type(self).vectors))

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
        """
        Check if this branch is branch labelled with ``label``.

        .. warning:

          Returns ``False`` even if all points are individually labelled with ``label``.
          Only when the branch itself is labelled will it return ``True``.

        :param label: The label to check for.
        :type label: str
        :rtype: boolean
        """
        return label in self._full_labels

    def has_any_label(self, labels):
        """
        Check if this branch is branch labelled with any of ``labels``.

        .. warning:

          Returns ``False`` even if all points are individually labelled with ``label``.
          Only when the branch itself is labelled will it return ``True``.

        :param labels: The labels to check for.
        :type labels: list
        :rtype: boolean
        """
        return any(self.has_label(l) for l in labels)

    def get_labelled_points(self, label):
        """
        Filter out all points with a certain label

        :param label: The label to check for.
        :type label: str
        :returns: All points with the label.
        :rtype: List[np.ndarray]
        """

        point_label_iter = zip(self.walk(), self.label_walk())
        return list(p for p, labels in point_label_iter if label in labels)

    def introduce_point(self, index, *args, labels=None):
        """
        Insert a new point at ``index``, before the existing point at ``index``.

        :param index: Index of the new point.
        :type index: int
        :param args: Vector coordinates of the new point
        :type args: float
        :param labels: The labels to assign to the point.
        :type labels: list
        """
        for v, vector_name in enumerate(type(self).vectors):
            vector = getattr(self, vector_name)
            new_vector = np.concatenate((vector[:index], [args[v]], vector[index:]))
            setattr(self, vector_name, new_vector)
        if labels is None:
            labels = set()
        for label, mask in self._label_masks.items():
            has_label = label in labels
            new_mask = np.concatenate((mask[:index], [has_label], mask[index:]))
            self._label_masks[label] = new_mask

    def introduce_arc_point(self, arc_val):
        """
        Introduce a new point at the given arc length.

        :param arc_val: Arc length between 0 and 1 to introduce new point at.
        :type arc_val: float
        :returns: The index of the new point.
        :rtype: int
        """
        arc = self.as_arc()
        arc_point_floor = self.floor_arc_point(arc_val)
        arc_point_ceil = self.ceil_arc_point(arc_val)
        arc_floor = arc[arc_point_floor]
        arc_ceil = arc[arc_point_ceil]
        point_floor = self[arc_point_floor]
        point_ceil = self[arc_point_ceil]
        rem = (arc_val - arc_floor) / (arc_ceil - arc_floor)
        new_point = (point_ceil - point_floor) * rem + point_floor
        new_index = arc_point_floor + 1
        self.introduce_point(new_index, *new_point)
        return new_index

    def get_arc_point(self, arc, eps=1e-10):
        """
        Strict search for an arc point within an epsilon.

        :param arc: Arclength position to look for.
        :type arc: float
        :param eps: Maximum distance/tolerance to accept an arc point as a match.
        :type eps: float
        :returns: The matched arc point index, or ``None`` if no match is found
        :rtype: Union[int, None]
        """
        arc_values = self.as_arc()
        arc_match = (i for i, arc_p in enumerate(arc_values) if abs(arc_p - arc) < eps)
        return next(arc_match, None)

    def as_matrix(self, with_radius=False):
        """
        Return the branch as a (PxV) matrix. The different vectors (V) are columns and
        each point (P) is a row.

        :param with_radius: Include the radius vector. Defaults to ``False``.
        :type with_radius: bool
        :returns: Matrix of the branch vectors.
        :rtype: :class:`numpy.ndarray`
        """
        # Get all the branch vectors unless not `with_radius`, then filter out `radii`.
        vector_names = (v for v in type(self).vectors if with_radius or v != "radii")
        vectors = list(getattr(self, name) for name in vector_names)
        return np.column_stack(vectors)

    def as_arc(self):
        """
        Return the branch as a vector of arclengths in the closed interval [0, 1]. An
        arclength is the distance each point to the start of the branch along the branch
        axis, normalized by total branch length. A point at the start will have an
        arclength close to 0, and a point near the end an arclength close to 1

        :returns: Vector of branch points as arclengths.
        :rtype: :class:`numpy.ndarray`
        """
        arc_distances = np.sqrt(np.sum(np.diff(self.as_matrix(), axis=0) ** 2, axis=1))
        arc_length = np.sum(arc_distances)
        return np.cumsum(np.concatenate(([0], arc_distances))) / arc_length

    def floor_arc_point(self, arc):
        """
        Get the index of the nearest proximal arc point.
        """
        p = 0
        for i, a in enumerate(self.as_arc()):
            if a <= arc:
                p = i
            else:
                break
        return p

    def ceil_arc_point(self, arc):
        """
        Get the index of the nearest distal arc point.
        """
        for i, a in enumerate(self.as_arc()):
            if a >= arc:
                return i
        return len(self) - 1

    def __getitem__(self, slice):
        return self.as_matrix(with_radius=True)[slice]


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
    """

    def __init__(self, roots):
        self.roots = roots

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
            # Empty morphology (or no branches with given labels)
            if matrix:
                return np.empty((0, len(vectors)))
            return tuple(np.empty(0) for _ in vectors)
        # Concatenate all of the branch vectors tail-to-head, store a tuple of
        # all the concatenated vector types.
        def concat_branches(v):
            return np.concatenate(tuple(getattr(b, v) for b in branches))

        t = tuple(concat_branches(v) for v in vectors)
        # Then optionally stack the vectors into a matrix or return the tuple
        return np.column_stack(t) if matrix else t

    def voxelize(self, N, labels=None):
        return VoxelCloud.create(self.get_branches(labels=labels), N)

    def get_bounding_box(self, labels=None, centered=True):
        # Should return a 0 based or soma centered bounding box from the
        # branches
        raise NotImplementedError("v4")

    def get_search_radius(self, plane="xyz"):
        raise NotImplementedError("Search radii should be replaced by Rtrees.")

    def get_plot_range(self, offset=[0.0, 0.0, 0.0]):
        raise NotImplementedError("Plotting should be factored out for deprecation.")
        compartments = self.compartment_tree.get_arrays()[0]
        n_dimensions = range(compartments.shape[1])
        mins = np.array([np.min(compartments[:, i]) + offset[i] for i in n_dimensions])
        max = np.max(
            np.array(
                [np.max(compartments[:, i]) - mins[i] + offset[i] for i in n_dimensions]
            )
        )
        return list(zip(mins.tolist(), (mins + max).tolist()))

    def rotate(self, v0, v):
        """

        Rotate a morphology to be oriented as vector v, supposing to start from orientation v0.
        norm(v) = norm(v0) = 1
        Rotation matrix R, representing a rotation of angle alpha around vector k

        """
        R = get_rotation_matrix(v0, v)
        raise NotImplementedError("Branch rotation")
        R.dot()


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
