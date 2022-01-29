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
import abc
import pickle
import h5py
import math
import itertools
import functools
import operator
import inspect
import numpy as np
from scipy.spatial.transform import Rotation
from ..voxels import VoxelSet
from ..exceptions import *
from ..reporting import report, warn
from .. import config


class MorphologySet:
    """
    Associates a set of :class:`StoredMorphologies
    <.storage.interfaces.StoredMorphology>` to cells
    """

    def __init__(self, loaders, m_indices):
        self._m_indices = m_indices
        self._loaders = loaders

    def __len__(self):
        return len(self._m_indices)

    def __iter__(self):
        return self.iter_morphologies()

    def get_indices(self):
        return self._m_indices

    def get(self, index, cache=True, hard_cache=False):
        data = self._m_indices[index]
        if data.ndim:
            if hard_cache:
                return np.array([self._loaders[idx].cached_load() for idx in data])
            elif cache:
                res = []
                for idx in data:
                    if idx not in self._cached:
                        self._cached[idx] = self._loaders[idx].load()
                    res.append(self._cached[idx].copy())
            else:
                res = np.array([self._loaders[idx].load() for idx in data])
            return res
        elif cache:
            if hard_cache:
                return self._loaders[data].cached_load()
            elif data not in self._cached:
                self._cached[data] = self._loaders[data].load()
            return self._cached[data]

    def clear_soft_cache(self):
        self._cached = {}

    def iter_morphologies(self, cache=True, unique=False, hard_cache=False):
        if unique:
            if hard_cache:
                yield from (l.cached_load() for l in self._loaders)
            else:
                yield from (l.load() for l in self._loaders)
        elif not cache:
            yield from (self._loaders[idx].load() for idx in self._m_indices)
        elif hard_cache:
            yield from (self._loaders[idx].cached_load() for idx in self._m_indices)
        else:
            _cached = {}
            for idx in self._m_indices:
                if idx not in _cached:
                    _cached[idx] = self._loaders[idx].load()
                yield _cached[idx].copy()

    def iter_meta(self, unique=False):
        if unique:
            yield from (l.get_meta() for l in self._loaders)
        else:
            yield from (self._loaders[idx].get_meta() for idx in self._m_indices)

    def _serialize_loaders(self):
        return [loader.get_meta()["name"] for loader in self._loaders]

    def merge(self, other):
        merge_offset = len(self._loaders)
        merged_loaders = self._loaders + other._loaders
        merged_indices = np.concatenate(
            (self._m_indices, other._m_indices + merge_offset)
        )
        return MorphologySet(merged_loaders, merged_indices)


class RotationSet:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return self.iter()

    def __getitem__(self, index):
        return np.fromiter((self._rot(d) for d in self._data[index]), dtype=Rotation)

    def iter(self, cache=False):
        if cache:
            yield from (self._cached_rot(tuple(d)) for d in self._data)
        else:
            yield from (self._rot(d) for d in self._data)

    @functools.cache
    def _cached_rot(self, angles):
        return self._rot(angles)

    def _rot(self, angles):
        return Rotation.from_euler("xyz", angles)


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


class SubTree:
    def __init__(self, branches, sanitize=True):
        if sanitize:
            # Find the roots of the full subtree(s) emanating from the given, possibly
            # overlapping branches.
            if len(branches) < 2:
                # The roots of the subtrees of 0 or 1 branches is eaqual to the 0 or 1
                # branches.
                self.roots = branches
            else:
                # Collect the deduplicated subtree emanating from all given branches
                sub = set(chain.from_iterable(b.get_branches() for b in branches))
                # Find the root branches whose parents are not part of the subtrees
                self.roots = [b for b in sub if b.parent not in sub]
        else:
            # No subtree sanitizing: Assume the whole tree is given, or only the roots
            # have been given, and just take all literal root (non-parent-having)
            # branches.
            self.roots = [b for b in branches if b.parent is None]

    @property
    def branches(self):
        """
        Return a depth-first flattened array of all branches.
        """
        return self.get_branches()

    def select(self, *labels):
        if not labels:
            labels = None
        return SubTree(self.get_branches(labels))

    def get_branches(self, labels=None):
        """
        Return a depth-first flattened array of all or the selected branches.

        :param labels: Names of the labels to select.
        :type labels: list
        :returns: List of all branches, or the ones fully labelled with any of
          the given labels.
        :rtype: list
        """
        root_iter = (branch_iter(root) for root in self.roots)
        all_branch = itertools.chain(*root_iter)
        if labels is None:
            return list(all_branch)
        else:
            return [b for b in all_branch if any(b.fully_labelled_as(l) for l in labels)]

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

    def rotate(self, rot, center=None):
        """
        Point rotation

        :param rot: Scipy rotation
        :type: :class:`scipy.spatial.transform.Rotation`
        """
        for b in self.branches:
            points = b.as_matrix(with_radius=False)
            if center is not None:
                points -= center
            rotated_points = rot.apply(points)
            if center is not None:
                rotated_points += center
            b.x, b.y, b.z = rotated_points.T

    def root_rotate(self, rot):
        """
        Rotate the subtree emanating from each root around the start of the root
        """
        for b in self.roots:
            group = SubTree([b])
            group.rotate(rot, group.origin)

    def translate(self, point):
        if len(point) != 3:
            raise ValueError("Point must be a sequence of x, y and z coordinates")
        for p, vector in zip(point, Branch.vectors):
            for branch in self.branches:
                array = getattr(branch, vector)
                array += p

    @property
    def origin(self):
        return np.mean([r.get_point(0) for r in self.roots], axis=0)

    def center(self):
        self.translate(-self.origin)

    def close_gaps(self):
        for branch in self.branches:
            if branch.parent is not None:
                gap_offset = branch.parent.get_point(-1) - branch.get_point(0)
                if not np.allclose(gap_offset, 0):
                    SubTree([branch]).translate(gap_offset)

    def collapse(self, on=None):
        if on is None:
            on = self.origin
        for root in self.roots:
            root.translate(on - root.get_point(0))

    def voxelize(self, N, labels=None):
        if labels is not None:
            raise NotImplementedError(
                "Can't voxelize labelled parts yet, require Selection API in morphologies.py, todo"
            )
        return VoxelSet.from_morphology(self, N)

    @functools.cache
    def cached_voxelize(self, N, labels=None):
        return self.voxelize(N, labels=labels)


class Morphology(SubTree):
    """
    A multicompartmental spatial representation of a cell based on a directed acyclic
    graph of branches whom consist of data vectors, each element of a vector being a
    coordinate or other associated data of a point on the branch.
    """

    def __init__(self, roots):
        super().__init__(roots, sanitize=False)
        if len(self.roots) < len(roots):
            warn("None-root branches given as morphology input.", MorphologyWarning)


def _copy_api(cls, wrap=lambda self: self):
    # Wraps functions so they are called with `self` wrapped in `wrap`
    def make_wrapper(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            return f(wrap(self), *args, **kwargs)

        return wrapper

    # Decorates a class so that it copies and wraps (see above) the public API of `cls`
    def decorator(decorated_cls):
        for key, f in vars(cls).items():
            if (
                inspect.isfunction(f)
                and not key.startswith("_")
                and key not in vars(decorated_cls)
            ):
                setattr(decorated_cls, key, make_wrapper(f))

        return decorated_cls

    return decorator


# For every `SubTree.f` there is a `Branch.f` == `SubTree([branch]).f` so we copy and wrap
# the public API of `SubTree` onto `Branch`, with the `SubTree([self])` wrapped into it.
@_copy_api(SubTree, lambda self: SubTree([self]))
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

    def __getitem__(self, slice):
        return self.as_matrix(with_radius=True)[slice]

    @property
    def parent(self):
        return self._parent

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

    @functools.wraps(SubTree.cached_voxelize)
    @functools.cache
    def cached_voxelize(self, *args, **kwargs):
        return SubTree([self]).voxelize(*args, **kwargs)


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
