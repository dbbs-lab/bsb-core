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

import functools
import inspect
import itertools
from collections import deque
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
from scipy.spatial.transform import Rotation

from .. import _util as _gutil
from .._encoding import EncodedLabels
from ..exceptions import EmptyBranchError, MorphologyDataError, MorphologyError
from ..voxels import VoxelSet


class MorphologySet:
    """
    Associates a set of :class:`StoredMorphologies
    <.storage.interfaces.StoredMorphology>` to cells
    """

    def __init__(self, loaders, m_indices=None, /, labels=None):
        """
        :param loaders: list of Morphology loader functions.
        :type loaders: List[Callable[[], bsb.storage.interfaces.StoredMorphology]]
        :param m_indices: indices of the loaders for each of the morphologies.
        :type: List[int]
        """
        if m_indices is None:
            loaders, m_indices = np.unique(loaders, return_inverse=True)
        self._m_indices = np.array(m_indices, copy=False, dtype=int)
        self._loaders = list(loaders)
        check_max = np.max(m_indices, initial=-1)
        if check_max >= len(loaders):
            raise IndexError(f"Index {check_max} out of range for {len(loaders)}.")
        self._cached = {}
        self._labels = labels

    def set_label_filter(self, labels):
        self._cached = {}
        for loader in self._loaders:
            loader._cached_load.cache_clear()
        self._labels = labels

    @_gutil.obj_str_insert
    def __repr__(self):
        return f"{len(self)} cells, {len(self._loaders)} morphologies"

    def __contains__(self, value):
        return value in [loader.name for loader in self._loaders]

    def count_morphologies(self):
        return len(self._loaders)

    def count_unique(self):
        uniques = []
        count = 0
        for m in (m.load() for m in self._loaders):
            if not any(f == m for f in uniques):
                uniques.append(m)
                count += 1
        return count

    def __len__(self):
        return len(self._m_indices)

    def __iter__(self):
        return self.iter_morphologies()

    @property
    def names(self):
        return [loader.name for loader in self._loaders]

    def get_indices(self, copy=True):
        return self._m_indices.copy() if copy else self._m_indices

    def get(self, index, cache=True, hard_cache=False):
        data = self._m_indices[index]
        if data.ndim:
            return self._get_many(data, cache, hard_cache)
        else:
            return self._get_one(data, cache, hard_cache)

    def _get_one(self, idx, cache, hard_cache):
        if cache:
            if hard_cache:
                return self._loaders[idx].cached_load(self._labels)

            if idx not in self._cached:
                self._cached[idx] = (
                    self._loaders[idx].load().set_label_filter(self._labels).as_filtered()
                )
            return self._cached[idx].copy()
        else:
            return self._loaders[idx].load().set_label_filter(self._labels).as_filtered()

    def _get_many(self, data, cache, hard_cache):
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
        if hard_cache:

            def _load(loader):
                return loader.cached_load(self._labels)

        elif self._labels is not None:

            def _load(loader):
                return loader.load().set_label_filter(self._labels).as_filtered()

        else:

            def _load(loader):
                return loader.load()

        if unique:
            yield from map(_load, self._loaders)
        elif not cache or hard_cache:
            yield from map(_load, (self._loaders[idx] for idx in self._m_indices))
        else:
            _cached = {}
            for idx in self._m_indices:
                if idx not in _cached:
                    _cached[idx] = _load(self._loaders[idx])
                yield _cached[idx].copy()

    def iter_meta(self, unique=False):
        if unique:
            yield from (loader.get_meta() for loader in self._loaders)
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

    def _mapback(self, locs):
        if self._labels is None:
            raise RuntimeError("Mapback requested on unfiltered morphology set.")
        locs = locs.copy()
        for i, loader in enumerate(self._loaders):
            rows = self._m_indices[locs[:, 0]] == i
            if np.any(rows):
                morpho = loader.load()
                filtered = morpho.set_label_filter(self._labels).as_filtered()
                # Using np.vectorize is Python speed O(n), worst case in numpy is C speed
                # O(n^2) (that is if every point is on another branch), not sure.
                branchmap = np.vectorize(
                    {
                        bid: b._copied_from_branch
                        for bid, b in enumerate(filtered.branches)
                    }.get
                )
                pointmap = np.vectorize(
                    {
                        bid: b._copied_points_offset
                        for bid, b in enumerate(filtered.branches)
                    }.get
                )
                # Map points first, then branches, since points depend on unmapped branch.
                locs[rows, 2] = locs[rows, 2] + pointmap(locs[rows, 1])
                locs[rows, 1] = branchmap(locs[rows, 1])
        return locs


class RotationSet:
    """
    Set of rotations. Returned rotations are of :class:`scipy.spatial.transform.Rotation`
    """

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            self._data = np.array(
                [
                    v.as_euler("xyz", degrees=True) if isinstance(v, Rotation) else v
                    for v in data
                ]
            )
        else:
            self._data = data
        if self._data.ndim != 2 or self._data.shape[1] != 3:
            raise ValueError("Input should be an (Nx3) matrix of rotations.")

    def __array__(self, dtype=None, *args, **kwargs):
        return self._data.__array__(dtype, *args, **kwargs)

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
    Collection of branches, not necessarily all connected.
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
        if not hasattr(self, "_is_shared"):
            raise UnpicklingError("Morphology class does not support pickling.")
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

    @points.setter
    def points(self, value):
        arr = np.array(value, copy=False, dtype=float)
        if self._is_shared:
            self.points[:] = arr
        else:
            ptr = 0
            for b in self.branches:
                b.points = arr[ptr : (ptr := ptr + len(b))]

    @property
    def radii(self):
        return self.flatten_radii()

    @radii.setter
    def radii(self, value):
        arr = np.array(value, copy=False, dtype=float)
        if self._is_shared:
            self.radii[:] = arr
        else:
            ptr = 0
            for b in self.branches:
                b.radii = arr[ptr : (ptr := ptr + len(b))]

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
        Return a dictionary containing mapping the id of the branch to its children.
        """
        idmap = {b: n for n, b in enumerate(self.branches)}
        return {n: list(map(idmap.get, b.children)) for n, b in enumerate(self.branches)}

    @property
    def path_length(self):
        """
        Return the total path length as the sum of the euclidian distances between
        consecutive points.
        """
        return sum(b.path_length for b in self.branches)

    def subtree(self, labels=None):
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
            return [b for b in all_branch if b.contains_labels(labels)]

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
            return np.empty(0)
        return np.concatenate(tuple(b.radii for b in branches))

    def flatten_labels(self):
        """
        Return the flattened labels of the morphology or subtree.

        :rtype: numpy.ndarray
        """
        if self._is_shared:
            return self._shared._labels
        else:
            return EncodedLabels.concatenate(*(b._labels for b in self.get_branches()))

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

    def label(self, labels, points=None):
        """
        Add labels to the morphology or subtree.

        :param labels: Labels to add to the subtree.
        :type labels: list[str]
        :param points: Optional boolean or integer mask for the points to be labelled.
        :type points: numpy.ndarray
        """
        if points is None:
            points = np.ones(len(self), dtype=bool)
        points = np.array(points, copy=False)
        if self._is_shared:
            self.labels.label(labels, points)
        else:
            if len(points) == len(self) and points.dtype == bool:
                ctr = 0
                for b in self.branches:
                    b.label(labels, points[ctr : ctr + len(b)])
                    ctr += len(b)
            elif np.can_cast(points.dtype, int):
                points = np.array(points, copy=False, dtype=int)
                for b in self.branches:
                    mux = points < len(b)
                    b.label(labels, points[mux])
                    points = points[~mux]
            else:
                raise ValueError(
                    "Label indices must be a boolean or integer mask."
                    f" {points.dtype} given."
                )

        return self

    def rotate(self, rotation, center=None):
        """
        Rotate the entire Subtree with respect to the center.
        The rotation angles are assumed to be in degrees.
        If the center is not provided, the Subtree will rotate from [0, 0, 0].

        :param rotation: Scipy rotation
        :type rotation: Union[scipy.spatial.transform.Rotation, List[float,float,float]]
        :param center: rotation offset point.
        :type center: numpy.ndarray
        """
        if not isinstance(rotation, Rotation):
            rotation = Rotation.from_euler("xyz", rotation, degrees=True)
        if self._is_shared:
            self._shared._points[:] = self._rotate(self._shared._points, rotation, center)
        else:
            for b in self.branches:
                b.points[:] = self._rotate(b.points, rotation, center)
        return self

    def _rotate(self, points, rot, center):
        if center is not None:
            points = points - center
            rotated_points = rot.apply(points)
            rotated_points = rotated_points + center
        else:
            rotated_points = rot.apply(points)
        return rotated_points

    def root_rotate(self, rot, downstream_of=0):
        """
        Rotate the subtree emanating from each root around the start of that root
        If downstream_of is provided, will rotate points starting from the index provided (only for
        subtrees with a single root).

        :param rot: Scipy rotation to apply to the subtree.
        :type rot: scipy.spatial.transform.Rotation
        :param downstream_of: index of the point in the subtree from which the rotation should be
            applied. This feature works only when the subtree has only one root branch.
        :returns: rotated Morphology
        :rtype: bsb.morphologies.SubTree
        """

        if downstream_of != 0:
            if len(self.roots) > 1:
                raise ValueError(
                    "Can't rotate with subbranch precision with multiple roots"
                )
            elif type(downstream_of) == int and 0 < downstream_of < len(
                self.roots[0].points
            ):
                b = self.roots[0]
                group = SubTree([b])
                upstream = np.copy(b.points[:downstream_of])
                group.rotate(rot, b.points[downstream_of])
                b.points[:downstream_of] = upstream
        else:
            for b in self.roots:
                group = SubTree([b])
                group.rotate(rot, group.origin)
        return self

    def translate(self, point):
        """
        Translate the subtree by a 3D vector.

        :param numpy.ndarray point: 3D vector to translate the subtree.
        :returns: the translated subtree
        :rtype: bsb.morphologies.SubTree
        """
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
        Collapse all the roots of the morphology or subtree onto a single point.

        :param on: Index of the root to collapse on. Collapses onto the origin by default.
        :type on: int
        """
        if on is None:
            on = self.origin
        for root in self.roots:
            root.translate(on - root.points[0])
        return self

    def simplify_branches(self, epsilon):
        """
        Apply Ramer–Douglas–Peucker algorithm to all points of all branches of the SubTree.
        :param epsilon: Epsilon to be used in the algorithm.
        """
        for branch in self.branches:
            branch.simplify(epsilon)

    def voxelize(self, N):
        """
        Turn the morphology or subtree into an approximating set of axis-aligned cuboids.

        :rtype: bsb.voxels.VoxelSet
        """
        return VoxelSet.from_morphology(self, N)

    @functools.cache
    def cached_voxelize(self, N):
        """
        Turn the morphology or subtree into an approximating set of axis-aligned cuboids
        and cache the result.

        :rtype: bsb.voxels.VoxelSet
        """
        return self.voxelize(N)


class _SharedBuffers:
    def __init__(self, points, radii, labels, properties):
        self._points = points
        self._radii = radii
        self._labels = labels if labels is not None else EncodedLabels.none(len(radii))
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

    def __init__(self, roots, meta=None, shared_buffers=None, sanitize=False):
        super().__init__(roots, sanitize=sanitize)
        self._meta = meta if meta is not None else {}
        self._filter = None
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

    @_gutil.obj_str_insert
    def __repr__(self):
        return (
            f"{len(self.roots)} roots, {len(self)} points,"
            f" from {self.bounds[0]} to {self.bounds[1]}"
        )

    def __eq__(self, other):
        return len(self.branches) == len(other.branches) and all(
            b1.is_terminal == b2.is_terminal and (not b1.is_terminal or b1 == b2)
            for b1, b2 in zip(self.branches, other.branches)
        )

    def __lt__(self, other):
        # Sorting compares using lt, so we use id for useless but stable comparison.
        return id(self) < id(other)

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
            labels = EncodedLabels.concatenate(*(b._labels for b in branches))
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

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def adjacency_dictionary(self):
        """
        Return a dictonary associating to each key (branch index) a list of adjacent branch indices
        """
        branches = self.branches
        idmap = {b: n for n, b in enumerate(branches)}
        return {n: list(map(idmap.get, b.children)) for n, b in enumerate(branches)}

    @property
    def labelsets(self):
        """
        Return the sets of labels associated to each numerical label.
        """
        self.optimize()
        return self._shared._labels.labels

    def list_labels(self):
        """
        Return a list of labels present on the morphology.
        """
        self.optimize()
        return sorted(set(_gutil.ichain(self._shared._labels.labels.values())))

    def set_label_filter(self, labels):
        """
        Set a label filter, so that `as_filtered` returns copies filtered by these labels.
        """
        self._filter = labels
        return self

    def get_label_mask(self, labels):
        """
        Get a mask corresponding to all the points labelled with 1 or more of the given
        labels
        """
        self.optimize()
        return self.labels.get_mask(labels)

    @classmethod
    def empty(cls):
        return cls([])

    def copy(self):
        """
        Copy the morphology.
        """
        # Make sure to optimize so that we use 1 shared buffer.
        self.optimize(force=False)
        # Copy that buffer into a new one
        buffers = self._shared.copy()
        roots = []
        branch_copy_map = {}
        ptr = 0
        # For each branch, create a copy, and assign a piece of the copied buffer to it.
        # Also attach it to its intended parent. Since we iterate DFS, each parent occurs
        # before their children, so we can always find it in `branch_copy_map` from a
        # previous iteration.
        for branch in self.branches:
            nptr = ptr + len(branch)
            nbranch = Branch(*buffers.get_shared(ptr, nptr))
            branch_copy_map[branch] = nbranch
            if not branch.is_root:
                branch_copy_map[branch.parent].attach_child(nbranch)
            else:
                roots.append(nbranch)
            ptr = nptr
        # Construct the morphology
        return self.__class__(roots, shared_buffers=buffers, meta=self.meta.copy())

    def as_filtered(self, labels=None):
        """
        Return a filtered copy of the morphology that includes only points that match the
        current label filter, or the specified labels.
        """
        filter = labels if labels is not None else self._filter
        if filter is None:
            return self.copy()
        self.optimize(force=False)
        buffers = self._shared.copy()
        roots = []
        branch_copy_map = {None: None}
        ptr = 0
        # Iterate over each branch, and turn it into 0 or more branches.
        for og_id, branch in enumerate(self.branches):
            # Using the filter mask, figure out where to split the branch into pieces.
            # Parts, or the entire branch, may be excluded if there are no labelled points
            filtered = branch.get_label_mask(filter)
            # Each boolean in filtered represents a point, either included or excluded.
            # Every subbranch begins where a point is excluded and the next point is
            # included, and ends where a point is included, and the next point is excluded
            starts = (np.nonzero(filtered[1:] & ~filtered[:-1])[0] + 1).tolist()
            ends = (np.nonzero(filtered[:-1] & ~filtered[1:])[0] + 1).tolist()
            # Treat the boundary.
            if len(filtered) and filtered[0]:
                starts.insert(0, 0)
            if len(filtered) and filtered[-1]:
                ends.append(len(filtered))
            prev = None
            nbranch = None
            # Make all the sub branches. Connect the first to the parent, and store the
            # last in the map, for children to be connected to.
            for start, end in zip(starts, ends):
                nbranch = Branch(*buffers.get_shared(ptr + start, ptr + end))
                # Store where this branch came from, for loc mapping.
                nbranch._copied_from_branch = og_id
                # Store where the points map to
                nbranch._copied_points_offset = start
                if not prev:
                    if branch.is_root or branch_copy_map[branch.parent] is None:
                        roots.append(nbranch)
                    else:
                        branch_copy_map[branch.parent].attach_child(nbranch)
                else:
                    prev.attach_child(nbranch)
                prev = nbranch
            ptr = ptr + len(branch)
            if nbranch is None:
                # If an entire branch is unlabelled, skip it, and map our children's
                # parent to their grandparent, since we, the parent, don't exist.
                branch_copy_map[branch] = branch_copy_map[branch.parent]
            else:
                # Did we create some branches? Use the last iteration value as parent for
                # our children.
                branch_copy_map[branch] = nbranch
        # Construct and return the morphology
        return self.__class__(roots, meta=self.meta.copy())

    def swap_axes(self, axis1: int, axis2: int):
        """
        Interchange two axes of a morphology points.

        :param int axis1: index of the first axis to exchange
        :param int axis2: index of the second axis to exchange
        :return: the modified morphology
        :rtype: bsb.morphologies.Morphology
        """
        if not 0 <= axis1 < 3 or not 0 <= axis2 < 3:
            raise ValueError(
                f"Axes values should be in [0, 1, 2], {axis1}, {axis2} given."
            )
        for b in self.branches:
            old_column = np.copy(b.points[:, axis1])
            b.points[:, axis1] = b.points[:, axis2]
            b.points[:, axis2] = old_column

        return self

    def simplify(self, *args, optimize=True, **kwargs):
        super().simplify_branches(*args, **kwargs)
        if optimize:
            self.optimize()

    def to_swc(self, file):
        """
        Create a SWC file from a Morphology.
        :param file: path to write to
        """
        file_data = _morpho_to_swc(self)
        if isinstance(file, str) or isinstance(file, Path):
            np.savetxt(
                file,
                file_data,
                fmt="%d %d %f %f %f %f %d",
                delimiter="\t",
                newline="\n",
                header="",
                footer="",
                comments="# ",
                encoding=None,
            )

    def to_graph_array(self):
        """
        Create a SWC-like numpy array from a Morphology.

        .. warning::

            Custom SWC tags (above 3) won't work and throw an error

        :returns: a numpy array with columns storing the standard SWC attributes
        :rtype: numpy.ndarray
        """
        data = _morpho_to_swc(self)
        return data


def _copy_api(cls, wrap=lambda self: self):
    # Wraps functions, so they are called with `self` wrapped in `wrap`
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

    def __init__(self, points, radii, labels=None, properties=None, children=None):
        """
        :param points: Array of 3D coordinates defining the point of the branch
        :type points: list | numpy.ndarray
        :param radii: Array of radii associated to each point
        :type radii: list | numpy.ndarray
        :param labels: Array of labels to associate to each point
        :type labels: List[str] | set | numpy.ndarray
        :param properties: dictionary of per-point data to store in the branch
        :type properties: dict
        :param children: list of child branches to attach to the branch
        :type children: List[bsb.morphologies.Branch]
        :raises bsb.exceptions.MorphologyError: if a property of the branch does not have the same
            size as its points
        """

        self._points = _gutil.sanitize_ndarray(points, (-1, 3), float)
        self._radii = _gutil.sanitize_ndarray(radii, (-1,), float)
        _gutil.assert_samelen(self._points, self._radii)
        self._children = []
        if labels is None:
            labels = EncodedLabels.none(len(points))
        elif not isinstance(labels, EncodedLabels):
            labels = EncodedLabels.from_labelset(len(points), labels)
        self._labels = labels
        if properties is None:
            properties = {}
        mismatched = [str(k) for k, v in properties.items() if len(v) != len(points)]
        if mismatched:
            raise MorphologyError(
                f"Morphology properties {', '.join(mismatched)} are not length {len(points)}"
            )
        self._properties = {
            k: v if isinstance(v, np.ndarray) else np.array(v)
            for k, v in properties.items()
        }
        self._parent = None
        self._on_mutate = lambda: None
        if children is not None:
            for child in children:
                self.attach_child(child)

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
        if not hasattr(self, "_properties"):
            raise UnpicklingError("Branch class does not support pickling.")
        if attr in self._properties:
            return self._properties[attr]
        else:
            super().__getattribute__(attr)

    def __copy__(self):
        return self.copy()

    def __bool__(self):
        # Without this, empty branches are False, and `if branch.parent:` checks fail.
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

    @points.setter
    def points(self, value):
        arr = np.array(value, copy=False, dtype=float)
        if arr.shape == self._points.shape:
            self._points[:] = arr
        elif arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Point data must have (N, 3) shape, {arr.shape} given.")
        else:
            self._points = arr

    @property
    def _kd_tree(self):
        """
        Return a `scipy.spatial.cKDTree` of this branch points for fast spatial queries.

        .. warning::

           Constructing a kd-tree takes time and should only be used for repeat queries.

        """
        import scipy.spatial

        return scipy.spatial.cKDTree(self._points)

    @property
    def point_vectors(self):
        """
        Return the individual vectors between consecutive points on this branch.
        """
        return np.diff(self.points, axis=0)

    @property
    def segments(self):
        """
        Return the start and end points of vectors between consecutive points on this
        branch.
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
        versor = (self.end - self.start) / np.linalg.norm(self.end - self.start)
        if np.any(np.isnan(versor)):
            raise EmptyBranchError("Empty and single-point branched have no versor")
        else:
            return versor

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
            raise EmptyBranchError(
                "Impossible to compute max_displacement in branches with 0 or 1 points."
            ) from None

    @property
    def path_length(self):
        """
        Return the sum of the euclidean distances between the points on the branch.
        """
        return np.sum(np.sqrt(np.sum(self.point_vectors**2, axis=1)))

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

    @radii.setter
    def radii(self, value):
        arr = np.array(value, copy=False, dtype=float)
        if arr.shape == self._radii.shape:
            self._radii[:] = arr
        else:
            self._radii = arr.ravel()

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

    def list_labels(self):
        """
        Return a list of labels present on the branch.
        """
        lookup = np.vectorize(self._labels.labels.get)
        labels = np.unique(lookup(self._labels.raw))
        return sorted(set(_gutil.ichain(labels)))

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
        props = {k: v.copy() for k, v in self._properties.items()}
        return cls(self._points.copy(), self._radii.copy(), self._labels.copy(), props)

    def label(self, labels, points=None):
        """
        Add labels to the branch.

        :param labels: Label(s) for the branch
        :type labels: List[str]
        :param points: An integer or boolean mask to select the points to label.
        """
        if points is None:
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

    def find_closest_point(self, coord):
        """
        Return the index of the closest on this branch to a desired coordinate.

        :param coord: The coordinate to find the nearest point to
        :type: :class:`numpy.ndarray`
        """
        diff = np.sqrt(np.sum((self._points - coord) ** 2, axis=1))
        return np.argmin(diff)

    def insert_branch(self, branch, index):
        """
        Split this branch and insert the given ``branch`` at the specified ``index``.

        :param branch: Branch to be attached
        :type branch: :class:`Branch <.morphologies.Branch>`
        :param index: Index or coordinates of the cutpoint; if coordinates are given, the closest point to the coordinates is used.
        :type: Union[:class:`numpy.ndarray`, int]
        """
        index = np.array(index, copy=False)
        if index.ndim != 0:
            index = self.find_closest_point(index)

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Cannot insert branch at cutpoint: index {index} is out of range ({len(self)})"
            )

        if index == len(self.points) - 1:
            self.attach_child(branch)
        elif index == 0:
            self.parent.attach_child(branch)
        else:
            first_segment = Branch(
                self._points.copy()[: index + 1],
                self._radii.copy()[: index + 1],
                self._labels.copy()[: index + 1],
                {k: v.copy()[: index + 1] for k, v in self._properties.items()},
            )
            self.parent.attach_child(first_segment)
            self.parent.detach_child(self)
            first_segment.attach_child(branch)
            second_segment = Branch(
                self._points.copy()[index:],
                self._radii.copy()[index:],
                self._labels.copy()[index:],
                {k: v.copy()[index:] for k, v in self._properties.items()},
            )
            for b in self.children:
                self.detach_child(b)
                second_segment.attach_child(b)
            first_segment.attach_child(second_segment)

    def detach(self):
        """
        Detach the branch from its parent, if one exists.
        """
        if self.parent:
            self.parent.detach_child(self)

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
            *self._properties.values(),
        )

    def contains_labels(self, labels):
        """
        Check if this branch contains any points labelled with any of the given labels.

        :param labels: The labels to check for.
        :type labels: List[str]
        :rtype: bool
        """
        return self.labels.contains(labels)

    def get_points_labelled(self, labels):
        """
        Filter out all points with certain labels

        :param labels: The labels to check for.
        :type labels: List[str] | numpy.ndarray[str]
        :returns: All points with the labels.
        :rtype: List[numpy.ndarray]
        """
        return self.points[self.get_label_mask(labels)]

    def get_label_mask(self, labels):
        """
        Return a mask for the specified labels

        :param labels: The labels to check for.
        :type labels: List[str] | numpy.ndarray[str]
        :returns: A boolean mask that selects out the points that match the label.
        :rtype: List[numpy.ndarray]
        """
        return self.labels.get_mask(labels)

    def introduce_point(self, index, position, radius=None, labels=None, properties=None):
        """
        Insert a new point at ``index``, before the existing point at ``index``.
        Radius, labels and extra properties can be set or will be copied from the
        existing point at ``index``.

        :param index: Index of the new point.
        :type index: int
        :param position: Coordinates of the new point
        :type position: List[float]
        :param radius: The radius to assign to the point.
        :type radius: float
        :param labels: The labels to assign to the point.
        :type labels: list
        :param properties: The properties to assign to the point.
        :type properties: dict
        """
        if index < 0 or index >= len(self.points):
            raise IndexError(
                f"Could not introduce point in branch at index {index}: out of bounds for branch length {len(self)}."
            )
        self._on_mutate()
        old_labels = self.labels[index]
        self.points = np.insert(self.points, index, position, 0)
        self._labels = np.insert(self._labels, index, old_labels)
        self._radii = np.insert(self._radii, index, radius or self._radii[index])
        # By default, duplicate the existing property value ...
        for k, v in self._properties.items():
            self._properties[k] = np.insert(v, index, v[index])
        if labels is not None:
            self.label(labels, [index])
        # ... and overwrite it with any new property values, if given.
        if properties is not None:
            for k, v in properties.items():
                if k in self._properties:
                    self._properties[k][index] = v
                else:
                    raise MorphologyError(
                        f"Property key '{k}' is not part of the Branch."
                    )

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

    def get_axial_distances(self, idx_start=0, idx_end=-1, return_max=False):
        """
        Return the displacements or its max value of a subset of branch points from its axis vector.
        :param idx_start = 0: index of the first point of the subset.
        :param idx_end = -1: index of the last point of the subset.
        :param return_max = False: if True the function only returns the max value of displacements, otherwise the entire array.
        """
        start = self.points[idx_start]
        end = self.points[idx_end]
        versor = (end - start) / np.linalg.norm(end - start)
        displacements = np.linalg.norm(
            np.cross(
                versor,
                (self.points[idx_start : idx_end + 1] - self.points[idx_start]),
            ),
            axis=1,
        )
        if return_max:
            try:
                return np.max(displacements)
            except IndexError:
                raise EmptyBranchError("Selected an empty subset of points") from None
        else:
            return displacements

    def delete_point(self, index):
        """
        Remove a point from the branch

        :param int index: index position of the point to remove
        :returns: the branch where the point has been removed
        :rtype: bsb.morphologies.Branch
        """
        self._points = np.delete(self._points, index, axis=0)
        self._labels = np.delete(self._labels, index, axis=0)
        self._radii = np.delete(self._radii, index, axis=0)
        for k, v in self._properties.items():
            self._properties[k] = np.delete(v, index, axis=0)
        return self

    def simplify(self, epsilon, idx_start=0, idx_end=-1):
        """
        Apply Ramer–Douglas–Peucker algorithm to all points or a subset of points of the branch.
        :param epsilon: Epsilon to be used in the algorithm.
        :param idx_start = 0: Index of the first element of the subset of points to be reduced.
        :param epsilon = -1: Index of the last element of the subset of points to be reduced.
        """
        if len(self.points) < 3:
            return
        if idx_end == -1:
            idx_end = len(self.points) - 1
        if epsilon < 0:
            raise ValueError(f"Epsilon must be >= 0")

        reduced = []
        skipped = deque()

        while True:
            dists = self.get_axial_distances(idx_start, idx_end)
            try:
                idx_max = np.argmax(dists)
                dmax = dists[idx_max]
                idx_max = idx_start + idx_max
            except ValueError:
                dmax = 0

            reduced.append(idx_start)
            reduced.append(idx_end)
            if dmax > epsilon and len(dists) > 2:
                skipped.append((idx_max, idx_end))
                idx_end = idx_max - 1
            else:
                try:
                    idx_start, idx_end = skipped.pop()
                except IndexError:
                    break

        # sorted because indexes are appended to reduced from the middle of the list (the first point with dist > epsilon)
        # then all points with smaller index  until 0, then all points with bigger index
        reduced = np.sort(np.unique(reduced))
        self.points = self.points[reduced]
        self.radii = self.radii[reduced]

    @functools.wraps(SubTree.cached_voxelize)
    @functools.cache
    def cached_voxelize(self, *args, **kwargs):
        return SubTree([self]).voxelize(*args, **kwargs)


def _morpho_to_swc(morpho):
    # Initialize an empty data array
    data = np.empty((len(morpho.points), 7), dtype=object)
    swc_tags = {"soma": 1, "axon": 2, "dendrites": 3}
    bmap = {}
    nid = 0
    offset = 0
    # Convert labels to tags
    if not hasattr(morpho, "tags"):
        tags = np.full(len(morpho.points), -1, dtype=int)
        for key in swc_tags.keys():
            mask = morpho.get_label_mask([key])
            tags[mask] = swc_tags[key]
    else:
        tags = morpho.tags
    if np.any(tags == -1):
        raise NotImplementedError("Can't store morphologies with custom SWC tags")
    # Iterate over the morphology branches
    for b in morpho.branches:
        ids = (
            np.arange(nid, nid + len(b) - 1)
            if len(b) > 1
            else np.arange(nid, nid + len(b))
        )
        samples = ids + 1
        data[ids, 0] = samples
        data[ids, 1] = tags[ids + offset]
        data[ids, 2:5] = morpho.points[ids + offset]
        try:
            data[ids, 5] = morpho.radii[ids + offset]
        except Exception as e:
            raise MorphologyDataError(
                f"Couldn't convert morphology radii to SWC: {e}."
                " Note that SWC files cannot store multi-dimensional radii"
            )
        nid += len(b) - 1 if len(b) > 1 else len(b)
        offset += 1
        bmap[b] = ids[-1]
        data[ids, 6] = ids
        data[ids[0], 6] = -1 if b.parent is None else bmap[b.parent] + 1

    return data[data != np.array(None)].reshape(-1, 7)


__all__ = [
    "Branch",
    "Morphology",
    "MorphologySet",
    "RotationSet",
    "SubTree",
    "branch_iter",
]
