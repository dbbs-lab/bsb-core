from ...interfaces import MorphologyRepository as IMorphologyRepository, StoredMorphology
from .resource import Resource
import numpy as np

_root = "/morphologies"


class MorphologyRepository(Resource, IMorphologyRepository):
    def __init__(self, engine):
        super().__init__(engine, _root)

    def select(self, *selectors):
        all_loaders = [self.preload_morphology(name) for name in self.keys()]
        selected = []
        for selector in selectors:
            selector.validate(all_loaders)
            selected.extend(filter(selector.pick, all_loaders))
        return selected

    def preload_morphology(self, name):
        return StoredMorphology(self._make_loader(name), self.get_meta(name))

    def _make_loader(self, name):
        def loader():
            return self.load_morphology(name)

        return loader

    def get_meta(self, name):
        return {"name": name}

    def load_morphology(self, name):
        with self._engine._read():
            with self._engine._handle("r") as f:
                try:
                    group = f[f"{self._path}/{name}/"]
                except:
                    raise MissingMorphologyError(
                        f"Repository at `{self._engine._root}` contains no morphology named `{name}`."
                    ) from None
                return _morphology(group)


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
