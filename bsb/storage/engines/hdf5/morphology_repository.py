from ....morphologies import Morphology, Branch
from ....exceptions import *
from ...interfaces import MorphologyRepository as IMorphologyRepository, StoredMorphology
from .resource import Resource
import numpy as np

_root = "/morphologies"


class MorphologyRepository(Resource, IMorphologyRepository):
    def __init__(self, engine):
        super().__init__(engine, _root)

    def select(self, *selectors):
        all_loaders = [self.preload(name) for name in self.keys()]
        selected = []
        for selector in selectors:
            selector.validate(all_loaders)
            selected.extend(filter(selector.pick, all_loaders))
        return selected

    def preload(self, name):
        return StoredMorphology(self._make_loader(name), self.get_meta(name))

    def _make_loader(self, name):
        def loader():
            return self.load(name)

        return loader

    # LOL, nice metadata
    def get_meta(self, name):
        return {"name": name}

    def has(self, name):
        with self._engine._read():
            with self._engine._handle("r") as repo:
                return f"{_root}/{name}" in repo

    def load(self, name):
        with self._engine._read():
            with self._engine._handle("r") as repo:
                try:
                    group = repo[f"{self._path}/{name}/"]
                except:
                    raise MissingMorphologyError(
                        f"Repository at `{self._engine._root}` contains no morphology named `{name}`."
                    ) from None
                return _morphology(group)

    def save(self, name, morphology, overwrite=False):
        with self._engine._write():
            with self._engine._handle("a") as repo:
                if self.has(name):
                    if overwrite:
                        self.remove(name)
                    else:
                        raise MorphologyRepositoryError(
                            f"A morphology called '{name}' already exists in this repository."
                        )
                root = repo["/morphologies"].create_group(name)
                branches_group = root.create_group("branches")
                for id, branch in enumerate(morphology.branches):
                    # Cheap trick: store the id assigned to each branch so that parent
                    # branches can give their id to their children for serialization.
                    branch._tmp_id = id
                    if branch._parent is not None:
                        parent_id = branch._parent._tmp_id
                    else:
                        parent_id = -1
                    self._save_branch(branches_group, id, branch, parent_id)
                points = morphology.flatten(["x", "y", "z"], matrix=True)
                root.attrs["lsb"] = np.min(points, axis=0)
                root.attrs["msb"] = np.max(points, axis=0)
        # Clean up the temporary ids stored on each branch.
        for branch in morphology.branches:
            del branch._tmp_id

    def _save_branch(self, group, branch_id, branch, parent_id):
        branch_group = group.create_group(str(branch_id))
        branch_group.attrs["parent"] = parent_id
        # Save vectors
        for v in Branch.vectors:
            branch_group.create_dataset(v, data=getattr(branch, v))
        # Save branch labels
        branch_group.attrs["branch_labels"] = branch._full_labels
        # Save point labels
        label_group = branch_group.create_group("labels")
        for label, label_mask in branch._label_masks.items():
            label_group.create_dataset(label, data=label_mask, dtype=np.bool)

    def remove(self, name):
        with self._engine._write():
            with self._engine._handle("a") as repo:
                try:
                    del repo[f"{_root}/{name}"]
                except KeyError:
                    raise MorphologyRepositoryError(f"'{name}' doesn't exist.") from None


def _morphology(m_root_group):
    b_root_group = m_root_group["branches"]
    branches = [_branch(b_group) for b_group in _int_ordered_iter(b_root_group)]
    _attach_branches(branches)
    roots = [b for b in branches if b._parent is None]
    morpho = Morphology(roots)
    # Until after rework a morphology still needs to know its name:
    morpho.morphology_name = m_root_group.name.split("/")[-1]
    return morpho


def _branch(b_root_group):
    vectors = _group_vector_iter(b_root_group, Branch.vectors)
    try:
        branch = Branch(*vectors)
    except KeyError:
        missing = [v for v in Branch.vectors if v not in b_root_group]
        raise MorphologyDataError(
            f"Missing branch vectors {missing} in '{b_root_group.name}'."
        )
    attrs = b_root_group.attrs
    branch._tmp_parent = int(attrs.get("parent", -1))
    if attrs.get("neuron_section", None) is not None:
        branch._neuron_sid = attrs.get("neuron_section")
    branch.label_all(*attrs.get("branch_labels", iter(())))
    for label, dataset in b_root_group["labels"].items():
        branch.label_points(label, dataset[()])
    return branch


def _attach_branches(branches):
    for branch in branches:
        if branch._tmp_parent < 0:
            continue
        branches[branch._tmp_parent].attach_child(branch)
        del branch._tmp_parent


def _group_vector_iter(group, vector_labels):
    return (group[label][()] for label in vector_labels)


def _is_invalid_order(order):
    # Checks sequential order starting from zero. [] is also valid.
    #
    # We need to prepend 0 to a 1 element diff so that 0 - 0 = len([0]) - 1 and all else
    # is rejected. `np.diff` behaves differently if `prepend` is set or not, there is no
    # default value that we can set that won't cause an error so we construct a dict and
    # either add the prepend kwarg to it or not and pass the dict as **kwargs.
    k = dict()
    if len(order) == 1:
        k["prepend"] = 0
    return bool(len(order) and np.sum(np.diff(order, **k)) != len(order) - 1)


def _int_ordered_iter(group):
    # Sort the group keys as ascending integers, then make sure they are a part of the
    # sequence [0, 1, 2, ..., n]
    try:
        neg = [*(g for g in group.keys() if int(g) < 0)]
    except ValueError:
        raise MorphologyDataError("Non numeric branch names are not allowed")
    if neg:
        raise MorphologyDataError(f"Branches with negative numbers {neg} are not allowed")
    order = sorted(map(int, group.keys()))
    if _is_invalid_order(order):
        raise MorphologyDataError(
            f"Non sequential branch numbering found: {order}. Branch numbers need to correspond with their index."
        )
    return (group[str(o)] for o in order)
