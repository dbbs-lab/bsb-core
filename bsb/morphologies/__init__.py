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
import inspect
import itertools
import functools
import operator
import inspect
import morphio
import numpy as np
from collections import deque
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
        self._cached = {}

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
            return self._cached[data].copy()

    def clear_soft_cache(self):
        self._cached = {}

    def iter_morphologies(self, cache=True, unique=False, hard_cache=False):
        """
        Iterate over the morphologies in a MorphologySet with full control over caching.

        :param cache: Use :ref:`soft-caching` (1 copy stored in mem per cache miss, 1 copy
          created from that per cache hit).
        :type cache: bool
        :param hard_cache: Use :ref:`hard-caching` (1 copy stored on the loader, always
          same copy returned from that loader forever).
        """
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
    """
    Set of rotations. Returned rotations are of :class:`scipy.spatial.transform.Rotation`
    """

    def __init__(self, data):
        self._data = data

    def __iter__(self):
        return self.iter()

    def __getitem__(self, index):
        data = self._data[index]
        if data.ndim == 2:
            return np.array([self._rot(d) for d in data])
        else:
            return self._rot(data)

    def __len__(self):
        return len(self._data)

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


class SubTree:
    """
    Collection of branches, not necesarily all connected.
    """

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
                sub = set(itertools.chain(*(b.get_branches() for b in branches)))
                # Find the root branches whose parents are not part of the subtrees
                self.roots = [b for b in sub if b.parent not in sub]
        else:
            # No subtree sanitizing: Assume the whole tree is given, or only the roots
            # have been given, and just take all literal root (non-parent-having)
            # branches.
            self.roots = [b for b in branches if b.parent is None]
        self._is_shared = False

    def __getattr__(self, attr):
        if self._is_shared:
            if attr in self._shared._prop:
                return self._shared._prop[attr]
            else:
                super().__getattribute__(attr)
        else:
            return np.concatenate([getattr(b, attr) for b in self.branches])

    def __len__(self):
        if self._is_shared:
            return len(self._shared._points)
        else:
            return sum(b.size for b in self.get_branches())

    @property
    def size(self):
        return len(self)

    @property
    def branches(self):
        """
        Return a depth-first flattened array of all branches.
        """
        return self.get_branches()

    @property
    def points(self):
        return self.flatten()

    @property
    def radii(self):
        return self.flatten_radii()

    @property
    def labels(self):
        return self.flatten_labels()

    @property
    def properties(self):
        return self.flatten_properties()

    @functools.cached_property
    def bounds(self):
        f = self.flatten()
        if not len(f):
            return np.zeros(3), np.zeros(3)
        return np.min(f, axis=0), np.max(f, axis=0)

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
            return [*all_branch]
        else:
            return [b for b in all_branch if b.contains_label(*labels)]

    def flatten(self):
        """
        Return the flattened points of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            return self._shared._points
        branches = self.get_branches()
        if not branches:
            return np.empty((0, 3))
        return np.vstack(tuple(b.points for b in branches))

    def flatten_radii(self):
        """
        Return the flattened radii of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            return self._shared._radii
        branches = self.get_branches()
        if not branches:
            return np.empty((0, 3))
        return np.vstack(tuple(b.radii for b in branches))

    def flatten_labels(self):
        """
        Return the flattened labels of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            return self._shared._labels
        else:
            raise NotImplementedError("todo")

    def flatten_properties(self):
        """
        Return the flattened properties of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            if not self._shared._prop:
                return np.empty((len(self), 0))
            return np.column_stack([*self._shared._prop.values()])
        else:
            raise NotImplementedError("todo")

    def rotate(self, rot, center=None):
        """
        Point rotation

        :param rot: Scipy rotation
        :type: :class:`scipy.spatial.transform.Rotation`
        """
        if self._is_shared:
            self._shared._points[:] = self._rotate(self._shared._points, rot, center)
        else:
            for b in self.branches:
                b.points[:] = self._rotate(b.points, rot, center)

    def _rotate(self, points, rot, center):
        if center is not None:
            points -= center
        rotated_points = rot.apply(points)
        if center is not None:
            rotated_points += center
        return rotated_points

    def root_rotate(self, rot):
        """
        Rotate the subtree emanating from each root around the start of that root
        """
        for b in self.roots:
            group = SubTree([b])
            group.rotate(rot, group.origin)

    def translate(self, point):
        if len(point) != 3:
            raise ValueError("Point must be a sequence of x, y and z coordinates")
        if self._is_shared:
            self._shared._points[:] += point
        else:
            for branch in self.branches:
                branch.points[:] += point

    @property
    def origin(self):
        return np.mean([r.get_point(0) for r in self.roots], axis=0)

    def center(self):
        """
        Center the morphology on the origin
        """
        self.translate(-self.origin)

    def close_gaps(self):
        """
        Close any head-to-tail gaps between parent and child branches.
        """
        for branch in self.branches:
            if branch.parent is not None:
                gap_offset = branch.parent.get_point(-1) - branch.get_point(0)
                if not np.allclose(gap_offset, 0):
                    SubTree([branch]).translate(gap_offset)

    def collapse(self, on=None):
        """
        Collapse all of the roots of the morphology or subtree onto a single point.

        :param on: Index of the root to collapse on. Collapses onto the origin by default.
        :type on: int
        """
        if on is None:
            on = self.origin
        for root in self.roots:
            root.translate(on - root.get_point(0))

    def voxelize(self, N, labels=None):
        """
        Turn the morphology or subtree into an approximating set of axis-aligned cuboids.

        :rtype: .voxels.VoxelSet
        """
        if labels is not None:
            raise NotImplementedError(
                "Can't voxelize labelled parts yet, require Selection API in morphologies.py, todo"
            )
        return VoxelSet.from_morphology(self, N)

    @functools.cache
    def cached_voxelize(self, N, labels=None):
        """
        Turn the morphology or subtree into an approximating set of axis-aligned cuboids
        and cache the result.

        :rtype: .voxels.VoxelSet
        """
        return self.voxelize(N, labels=labels)


class _SharedBuffers:
    def __init__(self, points, radii, labels, properties):
        self._points = points
        self._radii = radii
        self._labels = labels if labels is not None else _Labels.none(len(radii))
        self._prop = properties

    def copy(self):
        copied_props = {k: v.copy() for k, v in self._prop.items()}
        return self.__class__(
            self._points.copy(), self._radii.copy(), self._labels.copy(), copied_props
        )

    def points_shared(self, branches):
        return all(b.points.base is self._points for b in branches)

    def radii_shared(self, branches):
        return all(b.radii.base is self._radii for b in branches)

    def labels_shared(self, branches):
        return all(b.labels.base is self._labels for b in branches)

    def properties_shared(self, branches):
        return all(
            (
                b._properties.keys() == self._prop.keys() and all(c.base is self._prop[c])
                for a, c in b._properties.items()
            )
            for b in branches
        )

    def all_buffers_shared(self, branches):
        return (
            self.points_shared(branches)
            and self.radii_shared(branches)
            and self.labels_shared(branches)
            and self.properties_shared(branches)
        )

    def get_shared(self, start, end):
        copied_props = {k: v[start:end] for k, v in self._prop.items()}
        return (
            self._points[start:end],
            self._radii[start:end],
            self._labels[start:end],
            copied_props,
        )


class Morphology(SubTree):
    """
    A multicompartmental spatial representation of a cell based on a directed acyclic
    graph of branches whom consist of data vectors, each element of a vector being a
    coordinate or other associated data of a point on the branch.
    """

    def __init__(self, roots, meta=None, shared_buffers=None):
        super().__init__(roots, sanitize=False)
        if len(self.roots) < len(roots):
            warn("None-root branches given as morphology input.", MorphologyWarning)
        self._meta = meta if meta is not None else {}
        if shared_buffers is None:
            self._shared = None
            self._is_shared = False
        else:
            if isinstance(shared_buffers, _SharedBuffers):
                self._shared = shared_buffers
            else:
                self._shared = _SharedBuffers(*shared_buffers)
            self._is_shared = self._check_shared()
            for branch in self.branches:
                branch._on_mutate = self._mutnotif

    def _check_shared(self):
        if self._shared is None:
            return False
        return self._shared.all_buffers_shared(self.branches)

    def _mutnotif(self):
        self._is_shared = False

    @property
    def is_optimized(self):
        return self._shared

    def optimize(self):
        if not self._is_shared:
            raise NotImplementedError("todo")

    @property
    def meta(self):
        return self._meta

    def copy(self):
        self.optimize()
        buffers = self._shared.copy()
        roots = []
        branch_copy_map = {}
        bid = itertools.count()
        ptr = 0
        for branch in self.branches:
            nptr = ptr + len(branch)
            nbranch = Branch(*buffers.get_shared(ptr, nptr))
            branch_copy_map[branch] = nbranch
            if not branch.is_root:
                branch_copy_map[branch.parent].attach_child(nbranch)
            else:
                roots.append(nbranch)
        return self.__class__(roots, shared_buffers=buffers, meta=self.meta.copy())

    @classmethod
    def from_swc(cls, file, branch_class=None):
        """
        Create a Morphology from a file-like object.

        :param file: path or file-like object to parse.
        :type file: Union[str, TextIO]
        :param branch_class: Custom branch class
        :type branch_class: type
        :returns: The parsed morphology, with the SWC tags as a property.
        :rtype: bsb.morphologies.Morphology
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                return cls.from_swc(f, branch_class)
        if branch_class is None:
            branch_class = Branch
        return _swc_to_morpho(cls, branch_class, file.read())

    @classmethod
    def from_file(cls, path, branch_class=None):
        """
        Create a Morphology from a file on the file system through MorphIO.
        """
        if branch_class is None:
            branch_class = Branch
        return _import(cls, branch_class, path)


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

    def __init__(self, points, radii, labels=None, properties=None):
        self._points = points if isinstance(points, np.ndarray) else np.array(points)
        self._radii = radii if isinstance(radii, np.ndarray) else np.array(radii)
        self._children = []
        if labels is None:
            labels = _Labels.none(len(points))
        elif not isinstance(labels, _Labels):
            labels = _Labels.from_seq(len(points), labels)
        self._labels = labels
        if properties is None:
            properties = {}
        self._properties = {
            k: v if isinstance(v, np.ndarray) else np.array(v)
            for k, v in properties.items()
        }
        self._parent = None
        self._on_mutate = lambda: None

    def set_properties(self, **kwargs):
        for prop, values in kwargs.items():
            if len(values) != len(self):
                raise ValueError(f"Expected {len(self)} {prop}, got {len(values)}.")
            self._properties[prop] = values

    def __getattr__(self, attr):
        if attr in self._properties:
            return self._properties[attr]
        else:
            super().__getattribute__(attr)

    def __copy__(self):
        return self.copy()

    def __bool__(self):
        # Without this, empty branches are False, and messes with parent checking.
        return True

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
        return len(self._points)

    def __len__(self):
        return self.size

    @property
    def points(self):
        """
        Return the spatial coordinates of the points on this branch.
        """
        return self._points

    @property
    def radii(self):
        """
        Return the radii of the points on this branch.
        """
        return self._radii

    @property
    def labels(self):
        """
        Return the labels of the points on this branch. Labels are represented as a number
        that is associated to a set of labels. See :ref:`morphology_labels` for more info.
        """
        return self._labels

    @property
    def is_root(self):
        """
        Returns whether this branch is root or if it has a parent.

        :returns: True if this branch has no parent, False otherwise.
        :rtype: bool
        """
        return not self._parent

    @property
    def is_terminal(self):
        """
        Returns whether this branch is terminal or if it has children.

        :returns: True if this branch has no children, False otherwise.
        :rtype: bool
        """
        return not self._children

    def copy(self, branch_class=None):
        """
        Return a parentless and childless copy of the branch.

        :param branch_class: Custom branch creation class
        :type branch_class: type
        :returns: A branch, or `branch_class` if given, without parents or children.
        :rtype: bsb.morphologies.Branch
        """
        cls = branch_class or type(self)
        props = {k: v.copy() for k, v in self._properties}
        return cls(self._points.copy(), self._radii.copy(), self._labels.copy(), props)

    def label(self, *labels):
        """
        Add labels to every point on the branch. See
        :meth:`~.morphologies.Branch.label_points` to label individual points.

        :param labels: Label(s) for the branch. The first argument may also be a boolean
          or integer mask to select the points to label.
        :type labels: str
        """
        points = None
        if not labels:
            return
        elif not isinstance(labels[0], str):
            points = labels[0]
            labels = labels[1:]
        else:
            points = np.ones(len(self), dtype=bool)
        self._labels.label(labels, points)

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
        return zip(
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2],
            self.radii,
            self.labels.walk(),
            *self.properties.values(),
        )

    def contains_label(self, *labels):
        """
        Check if this branch is branch labelled with ``label``.

        .. warning:

          Returns ``False`` even if all points are individually labelled with ``label``.
          Only when the branch itself is labelled will it return ``True``.

        :param label: The label to check for.
        :type label: str
        :rtype: bool
        """
        return self.labels.contains(*labels)

    def has_any_label(self, labels):
        """
        Check if this branch is branch labelled with any of ``labels``.

        .. warning:

          Returns ``False`` even if all points are individually labelled with ``label``.
          Only when the branch itself is labelled will it return ``True``.

        :param labels: The labels to check for.
        :type labels: list
        :rtype: bool
        """
        return any(self.has_label(l) for l in labels)

    def get_labelled_points(self, label):
        """
        Filter out all points with a certain label

        :param label: The label to check for.
        :type label: str
        :returns: All points with the label.
        :rtype: List[numpy.ndarray]
        """
        return self.points[self.labels.get_mask(label)]

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
        self._on_mutate()
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

    def as_arc(self):
        """
        Return the branch as a vector of arclengths in the closed interval [0, 1]. An
        arclength is the distance each point to the start of the branch along the branch
        axis, normalized by total branch length. A point at the start will have an
        arclength close to 0, and a point near the end an arclength close to 1

        :returns: Vector of branch points as arclengths.
        :rtype: :class:`numpy.ndarray`
        """
        arc_distances = np.sqrt(np.sum(np.diff(self.points, axis=0) ** 2, axis=1))
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


class _Labels(np.ndarray):
    def __new__(subtype, *args, labels=None, **kwargs):
        kwargs["dtype"] = int
        obj = super().__new__(subtype, *args, **kwargs)
        if labels is None:
            labels = {0: set()}
        obj.labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.labels = getattr(obj, "labels", {0: set()})

    def label(self, labels, points):
        _transitions = {}
        counter = (c for c in itertools.count() if c not in self.labels)

        def transition(point):
            nonlocal _transitions
            if point in _transitions:
                return _transitions[point]
            else:
                trans_labels = self.labels[point].copy()
                trans_labels.update(labels)
                for k, v in self.labels.items():
                    if trans_labels == v:
                        return k
                else:
                    transition = next(counter)
                    self.labels[transition] = trans_labels
                    _transitions[point] = transition
                    return transition

        self[points] = np.vectorize(transition)(self[points])

    @classmethod
    def none(cls, len):
        return cls(len, buffer=np.zeros(len, dtype=int))

    @classmethod
    def from_seq(cls, len, seq):
        return cls(len, buffer=np.ones(len), labels={0: set(), 1: set(seq)})


def _swc_branch_dfs(adjacency, branches, node):
    branch = []
    branch_id = len(branches)
    branches.append((None, branch))
    node_stack = deque()
    while True:
        if node is not None:
            branch.append(node)
            child_nodes = adjacency[node]
        if not child_nodes:
            try:
                parent_bid, parent, node = node_stack.pop()
            except IndexError:
                break
            else:
                branch = [parent]
                branch_id = len(branches)
                branches.append((parent_bid, branch))
        elif len(child_nodes) == 1:
            node = child_nodes[0]
        else:
            node_stack.extend((branch_id, node, child) for child in reversed(child_nodes))
            child_nodes = []
            node = None


def _swc_to_morpho(cls, branch_cls, content):
    data = np.array(
        [
            swc_data
            for line in content.split("\n")
            if not line.strip().startswith("#")
            and (swc_data := [float(x) for x in line.split() if x != ""])
        ]
    )
    if data.dtype.name == "object":
        err_lines = ", ".join(i for i, d in enumerate(data) if len(d) != 7)
        raise ValueError(f"SWC incorrect on lines: {err_lines}")
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
        _swc_branch_dfs(adjacency, node_branches, root_node)
    branches = []
    roots = []
    _len = sum(len(s[1]) for s in node_branches)
    points = np.empty((_len, 3))
    radii = np.empty(_len)
    tags = np.empty(_len, dtype=int)
    labels = _Labels.none(_len)
    # Now turn each "node branch" into an actual branch by looking up the node data in the
    # samples array. We copy over the node data into several contiguous matrices that will
    # form the basis of the Morphology data structure.
    ptr = 0
    for parent, branch_nodes in node_branches:
        nptr = ptr + len(branch_nodes)
        node_data = data[branch_nodes]
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
            tags[ptr] = tags[ptr + 1]
        branch_tags = tags[ptr:nptr]
        # And the (empty) labels
        branch_labels = labels[ptr:nptr]
        ptr = nptr
        # Use the views to construct the branch
        branch = branch_cls(branch_points, branch_radii, branch_labels)
        branch.set_properties(tags=branch_tags)
        branches.append(branch)
        if parent is not None:
            branches[parent].attach_child(branch)
        else:
            roots.append(branch)
    # Then save the shared data matrices on the morphology
    morpho = cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}))
    # And assert that this shared buffer mode succeeded
    assert morpho._check_shared(), "SWC import didn't result in shareable buffers."
    return morpho


class Soma:
    def __init__(self, obj):
        self._o = obj

    def __getattr__(self, attr):
        return getattr(self._o, attr)


def _import(cls, branch_cls, file):
    morpho_io = morphio.Morphology(file)
    # We create shared buffers for the entire morphology, which optimize operations on the
    # entire morphology such as `.flatten`, subtree transformations and IO.  The branches
    # have views on those buffers, and as long as no points are added or removed, we can
    # keep working in shared buffer mode.
    soma = Soma(morpho_io.soma)
    _len = len(morpho_io.points) + len(soma.points)
    points = np.empty((_len, 3))
    radii = np.empty(_len)
    tags = np.empty(_len, dtype=int)
    labels = _Labels.none(_len)
    soma.children = morpho_io.root_sections
    section_stack = deque([(None, soma)])
    branch = None
    roots = []
    counter = itertools.count(1)
    ptr = 0
    while (i := next(counter)) :
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
            tags[ptr:nptr] = np.ones(len(section.points), dtype=int) * int(section.type)
            branch_tags = tags[ptr:nptr]
            branch_labels = labels[ptr:nptr]
            ptr = nptr
            # Pass the shared buffer views to the branch
            branch = branch_cls(branch_points, branch_radii, branch_labels)
            branch.set_properties(tags=branch_tags)
            if parent:
                parent.attach_child(branch)
            else:
                roots.append(branch)
            children = reversed([(branch, child) for child in section.children])
            section_stack.extend(children)
    morpho = cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}))
    assert morpho._check_shared(), "MorphIO import didn't result in shareable buffers."
    return morpho
