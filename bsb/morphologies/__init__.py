"""
Morphology module
"""

# This is a note to myself, should expand into docs:
#
# It goes ``morphology-on-file`` into ``repository`` that the ``storage`` needs to provide
# support for. Then after a placement job has placed cells for a chunk, the positions are
# sent to a ``distributor`` that is supposed to use the ``indicators`` to ask the
# ``storage.morphology_repository`` which ``loaders`` are appropriate for the given
# ``selectors``, then, still hopefully using just morpho metadata the  ``distributor``
# generates indices and rotations. In more complex cases the ``selector`` and
# ``distributor`` can both load the morphologies but this will slow things down.
#
# In the simulation step, these (possibly dynamically modified) morphologies are passed
# to the cell model instantiators.

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
from pathlib import Path
from scipy.spatial.transform import Rotation
from ..voxels import VoxelSet
from ..exceptions import *
from ..reporting import report, warn
from .. import config, _util as _gutil


class MorphologySet:
    """
    Associates a set of :class:`StoredMorphologies
    <.storage.interfaces.StoredMorphology>` to cells
    """

    def __init__(self, loaders, m_indices):
        self._m_indices = np.array(m_indices, copy=False)
        self._loaders = loaders
        check_max = np.max(m_indices, initial=-1)
        if check_max >= len(loaders):
            raise IndexError(f"Index {check_max} out of range for {len(loaders)}.")
        self._cached = {}

    def __contains__(self, value):
        return value in [l.name for l in self._loaders]

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
        return [loader.name for loader in self._loaders]

    @classmethod
    def empty(cls):
        return cls([], np.empty(0, dtype=int))

    def merge(self, other):
        merged_loaders = self._loaders.copy()
        previous_set = set(merged_loaders)
        if any(loader in previous_set for loader in other._loaders):
            # There is overlap between the sets, and mapping is required
            id_map = dict(
                (i, merged_loaders.index(loader))
                for i, loader in enumerate(other._loaders)
                if loader in previous_set
            )
            if all(k == v for k, v in id_map.items()):
                mapped_indices = other._m_indices
            else:

                def map_ids(id):
                    mapped_id = id_map.get(id, None)
                    if mapped_id is None:
                        mapped_id = id_map[id] = len(merged_loaders)
                        merged_loaders.append(other._loaders[id])
                    return mapped_id

                mapped_indices = np.vectorize(map_ids)(other._m_indices)
            merged_indices = np.concatenate((self._m_indices, mapped_indices))
        else:
            # No overlap, we can just offset the new dataset
            merge_offset = len(self._loaders)
            merged_loaders = self._loaders + other._loaders
            merged_indices = np.concatenate(
                (self._m_indices, other._m_indices + merge_offset)
            )
        return MorphologySet(merged_loaders, merged_indices)

    @classmethod
    def empty(cls):
        return cls([], np.empty(0, int))


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
                # branches themselves.
                self.roots = branches
            else:
                # Collect the deduplicated subtree emanating from all given branches; use
                # dict.fromkeys and .keys to preserve insertion order (i.e. DFS order)
                sub = dict.fromkeys(
                    itertools.chain(*(b.get_branches() for b in branches))
                ).keys()
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

    @property
    def bounds(self):
        f = self.flatten()
        if not len(f):
            return np.zeros(3), np.zeros(3)
        return np.min(f, axis=0), np.max(f, axis=0)

    @property
    def branch_adjacency(self):
        """
        Return a dictonary containing as items the children of the branch indexed by the key.
        """
        idmap = {b: n for n, b in enumerate(self.branches)}
        return {n: list(map(idmap.get, b.children)) for n, b in enumerate(self.branches)}

    def subtree(self, *labels):
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
            return _Labels.concatenate(*(b._labels for b in self.get_branches()))

    def flatten_properties(self):
        """
        Return the flattened properties of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            if not self._shared._prop:
                return {}
            return self._shared._prop.copy()
        else:
            branches = self.get_branches()
            len_ = sum(len(b) for b in branches)
            all_props = [*set(_gutil.ichain(b._properties.keys() for b in branches))]
            props = {k: np.empty(len_) for k in all_props}
            ptr = 0
            for branch in self.branches:
                nptr = ptr + len(branch)
                for k, v in props.items():
                    prop = branch._properties.get(k, None)
                    if prop is None:
                        prop = np.full(len(branch), np.nan)
                    v[ptr:nptr] = prop
                ptr = nptr
            return props

    def label(self, *labels):
        """
        Add labels to the morphology or subtree.

        :param labels: Label(s) for the branch. The first argument may also be a boolean
          or integer mask to select the points to label.
        :type labels: str
        """
        if labels:
            if self._is_shared:
                if not isinstance(labels[0], str):
                    points = labels[0]
                    labels = labels[1:]
                else:
                    points = np.ones(len(self), dtype=bool)
                self._labels.label(labels, points)
            else:
                for b in self.branches:
                    b.label(*labels)
        return self

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
        return self

    def _rotate(self, points, rot, center):
        if center is not None:
            points = points - center
            rotated_points = rot.apply(points)
            rotated_points = rotated_points + center
        else:
            rotated_points = rot.apply(points)
        return rotated_points

    def root_rotate(self, rot):
        """
        Rotate the subtree emanating from each root around the start of that root
        """
        for b in self.roots:
            group = SubTree([b])
            group.rotate(rot, group.origin)
        return self

    def translate(self, point):
        if len(point) != 3:
            raise ValueError("Point must be a sequence of x, y and z coordinates")
        if self._is_shared:
            self._shared._points[:] = self._shared._points + point
        else:
            for branch in self.branches:
                branch.points[:] = branch.points[:] + point
        return self

    @property
    def origin(self):
        return np.mean([r.points[0] for r in self.roots], axis=0)

    def center(self):
        """
        Center the morphology on the origin
        """
        self.translate(-self.origin)
        return self

    def close_gaps(self):
        """
        Close any head-to-tail gaps between parent and child branches.
        """
        for branch in self.branches:
            if branch.parent is not None:
                gap_offset = branch.parent.points[-1] - branch.points[0]
                if not np.allclose(gap_offset, 0):
                    SubTree([branch]).translate(gap_offset)
        return self

    def collapse(self, on=None):
        """
        Collapse all of the roots of the morphology or subtree onto a single point.

        :param on: Index of the root to collapse on. Collapses onto the origin by default.
        :type on: int
        """
        if on is None:
            on = self.origin
        for root in self.roots:
            root.translate(on - root.points[0])
        return self

    def voxelize(self, N, labels=None):
        """
        Turn the morphology or subtree into an approximating set of axis-aligned cuboids.

        :rtype: bsb.voxels.VoxelSet
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

        :rtype: bsb.voxels.VoxelSet
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

    def __eq__(self, other):
        return len(self.branches) == len(other.branches) and all(
            b1.is_terminal == b2.is_terminal and (not b1.is_terminal or b1 == b2)
            for b1, b2 in zip(self.branches, other.branches)
        )

    def __hash__(self):
        return id(self)

    def _check_shared(self):
        if self._shared is None:
            return False
        return self._shared.all_buffers_shared(self.branches)

    def _mutnotif(self):
        self._is_shared = False

    @property
    def is_optimized(self):
        return self._shared

    def optimize(self, force=False):
        if force or not self._is_shared:
            branches = self.branches
            len_ = sum(len(b) for b in branches)
            points = np.empty((len_, 3))
            radii = np.empty(len_)
            all_props = [*set(_gutil.ichain(b._properties.keys() for b in branches))]
            types = [
                next(_p[k].dtype for b in branches if k in (_p := b._properties))
                for k in all_props
            ]
            props = {k: np.empty(len_, dtype=t) for k, t in zip(all_props, types)}
            labels = _Labels.concatenate(*(b._labels for b in branches))
            ptr = 0
            for branch in self.branches:
                nptr = ptr + len(branch)
                points[ptr:nptr] = branch.points
                branch._points = points[ptr:nptr]
                radii[ptr:nptr] = branch.radii
                branch._radii = radii[ptr:nptr]
                for k, v in props.items():
                    prop = branch._properties.get(k, None)
                    if prop is None:
                        prop = np.full(len(branch), np.nan)
                    v[ptr:nptr] = prop
                    branch._properties[k] = v[ptr:nptr]
                branch._labels = labels[ptr:nptr]
                ptr = nptr
            self._shared = _SharedBuffers(points, radii, labels, props)
            self._is_shared = True
            assert self._check_shared(), "optimize should result in shared buffers"

    @property
    def meta(self):
        return self._meta

    @property
    def labelsets(self):
        """
        Return the sets of labels associated to each numerical label.
        """
        self.optimize()
        return self._shared._labels.labels

    def copy(self):
        self.optimize(force=False)
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
    def from_swc(cls, file, branch_class=None, tags=None, meta=None):
        """
        Create a Morphology from a file-like object.

        :param file: path or file-like object to parse.
        :param branch_class: Custom branch class
        :type branch_class: type
        :returns: The parsed morphology, with the SWC tags as a property.
        :rtype: bsb.morphologies.Morphology
        """
        if isinstance(file, str) or isinstance(file, Path):
            with open(str(file), "r") as f:
                return cls.from_swc(f, branch_class, meta=meta)
        if branch_class is None:
            branch_class = Branch
        return _swc_to_morpho(cls, branch_class, file.read(), tags=tags, meta=meta)

    @classmethod
    def from_file(cls, path, branch_class=None, meta=None):
        """
        Create a Morphology from a file on the file system through MorphIO.
        """
        if branch_class is None:
            branch_class = Branch
        return _import(cls, branch_class, path, meta=meta)

    @classmethod
    def from_arbor(cls, arb_m, centering=True, branch_class=None, meta=None):
        if branch_class is None:
            branch_class = Branch
        return _import_arb(cls, arb_m, centering, branch_class, meta=meta)

    def to_swc(self, file, meta=None):
        """
        Create a SWC file from a Morphology.
        :param file: path or file-like object to parse.
        :param branch_class: Custom branch class
        """
        file_data = _morpho_to_swc(self)
        if meta:  # pragma: nocover
            raise NotImplementedError(
                "Can't store morpho header yet, require special handling in morphologies/__init__.py, todo"
            )

        if isinstance(file, str) or isinstance(file, Path):
            np.savetxt(
                file,
                file_data,
                fmt=f"%d %d %f %f %f %f %d",
                delimiter="\t",
                newline="\n",
                header="",
                footer="",
                comments="# ",
                encoding=None,
            )


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
            labels = _Labels.from_labelset(len(points), labels)
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
            if prop in self._properties:
                self._properties[prop][:] = values
            else:
                self._on_mutate()
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

    def __eq__(self, other):
        if isinstance(other, Branch):
            return (
                self.points.shape == other.points.shape
                and np.allclose(self.points, other.points)
                and self.labels == other.labels
            )
        else:
            return np.allclose(self.points, other)

    def __hash__(self):
        return id(self)

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
    def point_vectors(self):
        """
        Return the individual vectors between consecutive points on this branch.
        """
        return np.diff(self.points, axis=0)

    @property
    def segments(self):
        """
        Return the start and end points of vectors between consecutive points on this branch.
        """
        return np.hstack(
            (self.points[:-1], self.points[:-1] + self.point_vectors)
        ).reshape(-1, 2, 3)

    @property
    def start(self):
        """
        Return the spatial coordinates of the starting point of this branch.
        """
        try:
            return self._points[0]
        except IndexError:
            raise EmptyBranchError("Empty branch has no starting point") from None

    @property
    def end(self):
        """
        Return the spatial coordinates of the terminal point of this branch.
        """
        try:
            return self._points[-1]
        except IndexError:
            raise EmptyBranchError("Empty branch has no ending point") from None

    @property
    def vector(self):
        """
        Return the vector of the axis connecting the start and terminal points.
        """
        try:
            return self.end - self.start
        except IndexError:
            raise EmptyBranchError("Empty branch has no vector") from None

    @property
    def versor(self):
        """
        Return the normalized vector of the axis connecting the start and terminal points.
        """
        try:
            return (self.end - self.start) / np.linalg.norm(self.end - self.start)
        except IndexError:
            raise EmptyBranchError("Empty branch has no versor") from None

    @property
    def euclidean_dist(self):
        """
        Return the Euclidean distance from the start to the terminal point of this branch.
        """
        try:
            return np.sqrt(np.sum((self.end - self.start) ** 2))
        except IndexError:
            raise EmptyBranchError("Empty branch has no Euclidean distance") from None

    @property
    def path_dist(self):
        """
        Return the path distance from the start to the terminal point of this branch,
        computed as the sum of Euclidean segments between consecutive branch points.
        """
        try:
            return np.sum(np.sqrt(np.sum(self.point_vectors**2, axis=1)))
        except IndexError:
            raise EmptyBranchError("Empty branch has no path distance") from None

    @property
    def max_displacement(self):
        """
        Return the max displacement of the branch points from its axis vector.
        """
        try:
            displacements = np.linalg.norm(
                np.cross(self.versor, (self.points - self.start)), axis=1
            )
            return np.max(displacements)
        except IndexError:
            raise EmptyBranchError("Empty branch has no displaced points") from None

    @property
    def fractal_dim(self):
        """
        Return the fractal dimension of this branch, computed as the coefficient
        of the line fitting the log-log plot of path vs euclidean distances of its points.
        """
        if len(self.points) == 0:
            raise EmptyBranchError("Empty branch has no fractal dimension") from None
        else:
            euclidean = np.sqrt(np.sum((self.points - self.start) ** 2, axis=1))
            path = np.cumsum(np.sqrt(np.sum(self.point_vectors**2, axis=1)))
            log_e = np.log(euclidean[1:])
            log_p = np.log(path)
            if len(self.points) <= 2:
                return 1.0
            return np.polyfit(log_e, log_p, 1)[0]

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
    def labelsets(self):
        """
        Return the sets of labels associated to each numerical label.
        """
        return self._labels.labels

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
        Add labels to the branch.

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
        self._on_mutate()
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
        if branch._parent is not self:
            raise ValueError(f"Can't detach {branch} from {self}, not a child branch.")
        self._on_mutate()
        self._children = [b for b in self._children if b is not branch]
        branch._parent = None

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
        Check if this branch contains any points labelled with any of the given labels.

        :param labels: The labels to check for.
        :type labels: List[str]
        :rtype: bool
        """
        return self.labels.contains(*labels)

    def get_points_labelled(self, label):
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


class _lset(set):
    def __hash__(self):
        return int.from_bytes(":|\!#è".join(sorted(self)).encode(), "little")

    def copy(self):
        return self.__class__(self)


class _Labels(np.ndarray):
    def __new__(subtype, *args, labels=None, **kwargs):
        kwargs["dtype"] = int
        obj = super().__new__(subtype, *args, **kwargs)
        if labels is None:
            labels = {0: _lset()}
        if any(not isinstance(v, _lset) for v in labels.values()):
            labels = {k: _lset(v) for k, v in labels.items()}
        obj.labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.labels = getattr(obj, "labels", {0: _lset()})

    def __eq__(self, other):
        return np.allclose(*_Labels._merged_translate((self, other)))

    def copy(self, *args, **kwargs):
        cp = super().copy(*args, **kwargs)
        cp.labels = {k: v.copy() for k, v in cp.labels.items()}
        return cp

    def label(self, labels, points):
        _transitions = {}
        # A counter that skips existing values.
        counter = (c for c in itertools.count() if c not in self.labels)

        # This local function looks up the new id that a point should transition
        # to when `labels` are added to the labels it already has.
        def transition(point):
            nonlocal _transitions
            # Check if we already know the transition of this value.
            if point in _transitions:
                return _transitions[point]
            else:
                # First time making this transition. Join the existing and new labels
                trans_labels = self.labels[point].copy()
                trans_labels.update(labels)
                # Check if this new combination of labels already is assigned an id.
                for k, v in self.labels.items():
                    if trans_labels == v:
                        # Transition labels already exist, return it
                        return k
                else:
                    # Transition labels are a new combination, store them under a new id.
                    transition = next(counter)
                    self.labels[transition] = trans_labels
                    # Cache the result
                    _transitions[point] = transition
                    return transition

        # Replace the label values with the transition values
        self[points] = np.vectorize(transition)(self[points])

    def contains(self, *labels):
        return np.any(self.get_mask(*labels))

    def get_mask(self, *labels):
        has_any = [k for k, v in self.labels.items() if any(l in v for l in labels)]
        return np.isin(self, has_any)

    def walk(self):
        """
        Iterate over the branch, yielding the labels of each point.
        """
        for x in self:
            yield self.labels[x].copy()

    def expand(self, label):
        """
        Translate a label value into its corresponding labelset.
        """
        return self.labels[label].copy()

    @classmethod
    def none(cls, len):
        """
        Create _Labels without any labelsets.
        """
        return cls(len, buffer=np.zeros(len, dtype=int))

    @classmethod
    def from_labelset(cls, len, labelset):
        """
        Create _Labels with all points labelled to the given labelset.
        """
        return cls(len, buffer=np.ones(len), labels={0: _lset(), 1: _lset(seq)})

    @staticmethod
    def _get_merged_lookups(arrs):
        if not arrs:
            return {0: _lset()}
        merged = {}
        new_labelsets = set()
        to_map_arrs = {}
        for arr in arrs:
            for k, l in arr.labels.items():
                if k not in merged:
                    # The label spot is available, so take it
                    merged[k] = l
                elif merged[k] != l:
                    # The labelset doesn't match, so this array will have to be mapped,
                    # and a new spot found for the conflicting labelset.
                    new_labelsets.add(l)
                    # np ndarray unhashable, for good reason, so use `id()` for quick hash
                    to_map_arrs[id(arr)] = arr
                # else: this labelset matches with the superset's nothing to do

        # Collect new spots for new labelsets
        counter = (c for c in itertools.count() if c not in merged)
        lset_map = {}
        for labelset in new_labelsets:
            key = next(counter)
            merged[key] = labelset
            lset_map[labelset] = key

        return merged, to_map_arrs, lset_map

    def _merged_translate(arrs, lookups=None):
        if lookups is None:
            merged, to_map_arrs, lset_map = _Labels._get_merged_lookups(arrs)
        else:
            merged, to_map_arrs, lset_map = lookups
        for arr in arrs:
            if id(arr) not in to_map_arrs:
                # None of the label array's labelsets need to be mapped, good as is.
                block = arr
            else:
                # Lookup each labelset, if found, map to new value, otherwise, map to
                # original value.
                arrmap = {og: lset_map.get(lset, og) for og, lset in arr.labels.items()}
                block = np.vectorize(arrmap.get)(arr)
            yield block

    @classmethod
    def concatenate(cls, *label_arrs):
        if not label_arrs:
            return _Labels.none(0)
        lookups = _Labels._get_merged_lookups(label_arrs)
        total = sum(len(l) for l in label_arrs)
        concat = cls(total, labels=lookups[0])
        ptr = 0
        for block in _Labels._merged_translate(label_arrs, lookups):
            nptr = ptr + len(block)
            # Concatenate the translated block
            concat[ptr:nptr] = block
            ptr = nptr
        return concat


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


def _swc_to_morpho(cls, branch_cls, content, tags=None, meta=None):
    tag_map = {1: "soma", 2: "axon", 3: "dendrites"}
    if tags is not None:
        tag_map.update(tags)
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
            # Since we add an extra point we have to copy its tag from the next point.
            tags[ptr] = tags[ptr + 1]
        branch_tags = tags[ptr:nptr]
        # And the labels
        branch_labels = labels[ptr:nptr]
        for v in np.unique(branch_tags):
            branch_labels.label([tag_map.get(v, f"tag_{v}")], branch_tags == v)
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
    morpho = cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}), meta=meta)
    # And assert that this shared buffer mode succeeded
    assert morpho._check_shared(), "SWC import didn't result in shareable buffers."
    return morpho


def _morpho_to_swc(morpho):
    # Initialize an empty data array
    data = np.empty((len(morpho.points), 7), dtype=object)
    bmap = {}
    nid = 0
    # Iterate over the morphology branches
    for b in morpho.branches:
        ids = (
            np.arange(nid, nid + len(b) - 1)
            if len(b) > 1
            else np.arange(nid, nid + len(b))
        )
        if len(b.labelsets.keys()) > 4:  # pragma: nocover
            # Standard labels are 0,1,2,3
            raise NotImplementedError(
                "Can't store custom labelled nodes yet, require special handling in morphologies/__init__.py, todo"
            )
        samples = ids + 1
        data[ids, 0] = samples
        data[ids, 1] = b.labels[1:] if len(b) > 1 else b.labels
        data[ids, 2:5] = b.points[1:] if len(b) > 1 else b.points
        try:
            data[ids, 5] = b.radii[1:] if len(b) > 1 else b.radii
        except Exception as e:
            raise MorphologyDataError(
                f"Couldn't convert morphology radii to SWC: {e}. Note that SWC files cannot store multi-dimensional radii"
            )
        nid += len(b) - 1 if len(b) > 1 else len(b)
        bmap[b] = ids[-1]
        data[ids, 6] = ids
        data[ids[0], 6] = -1 if b.parent is None else bmap[b.parent] + 1

    return data[data != np.array(None)].reshape(-1, 7)


# Wrapper to append our own attributes to morphio somas and treat it like any other branch
class _MorphIoSomaWrapper:
    def __init__(self, obj):
        self._o = obj

    def __getattr__(self, attr):
        return getattr(self._o, attr)


def _import(cls, branch_cls, file, meta=None):
    morpho_io = morphio.Morphology(file)
    # We create shared buffers for the entire morphology, which optimize operations on the
    # entire morphology such as `.flatten`, subtree transformations and IO.  The branches
    # have views on those buffers, and as long as no points are added or removed, we can
    # keep working in shared buffer mode.
    soma = _MorphIoSomaWrapper(morpho_io.soma)
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
    while i := next(counter):
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
    morpho = cls(roots, shared_buffers=(points, radii, labels, {"tags": tags}), meta=meta)
    assert morpho._check_shared(), "MorphIO import didn't result in shareable buffers."
    return morpho


def _import_arb(cls, arb_m, centering, branch_class, meta=None):
    import arbor

    decor = arbor.decor()
    morpho_roots = set(
        i for i in range(arb_m.num_branches) if arb_m.branch_parent(i) == 4294967295
    )
    root_prox = [r[0].prox for r in map(arb_m.branch_segments, morpho_roots)]
    center = np.mean([[p.x, p.y, p.z] for p in root_prox], axis=0)
    parent = None
    roots = []
    stack = []
    cable_id = morpho_roots.pop()
    while True:
        segments = arb_m.branch_segments(cable_id)
        if not segments:
            branch = Branch([], [], [], [])
        else:
            # Prepend the proximal end of the first segment to get [p0, p1, ..., pN]
            x = np.array([segments[0].prox.x] + [s.dist.x for s in segments])
            y = np.array([segments[0].prox.y] + [s.dist.y for s in segments])
            z = np.array([segments[0].prox.z] + [s.dist.z for s in segments])
            r = np.array([segments[0].prox.radius] + [s.dist.radius for s in segments])
            if centering:
                x -= center[0]
                y -= center[1]
                z -= center[2]
            branch = branch_class(x, y, z, r)
        branch._cable_id = cable_id
        if parent:
            parent.attach_child(branch)
        else:
            roots.append(branch)
        children = arb_m.branch_children(cable_id)
        if children:
            stack.extend((branch, child) for child in reversed(children))
        if stack:
            parent, cable_id = stack.pop()
        elif not morpho_roots:
            break
        else:
            parent = None
            cable_id = morpho_roots.pop()

    morpho = cls(roots, meta=meta)
    branches = morpho.branches
    branch_map = {branch._cable_id: branch for branch in branches}
    cc = arbor.cable_cell(arb_m, labels, decor)
    for label in labels:
        if "excl:" in label or label == "all":
            continue
        label_cables = cc.cables(f'"{label}"')
        for cable in label_cables:
            cable_id = cable.branch
            branch = branch_map[cable_id]
            if cable.dist == 1 and cable.prox == 0:
                branch.label(label)
            else:
                prox_index = branch.get_arc_point(cable.prox, eps=1e-7)
                if prox_index is None:
                    prox_index = branch.introduce_arc_point(cable.prox)
                dist_index = branch.get_arc_point(cable.dist, eps=1e-7)
                if dist_index is None:
                    dist_index = branch.introduce_arc_point(cable.dist)
                mask = np.array(
                    [False] * prox_index
                    + [True] * (dist_index - prox_index + 1)
                    + [False] * (len(branch) - dist_index - 1)
                )
                branch.label(mask, label)

    morpho.optimize()
    return morpho
