from ....morphologies import Morphology, Branch
from ....exceptions import *
from ...interfaces import MorphologyRepository as IMorphologyRepository, StoredMorphology
from .resource import Resource
import numpy as np
import arbor

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
        return StoredMorphology(name, self._make_loader(name), self.get_meta(name))

    def _make_loader(self, name):
        def loader():
            return self.load(name)

        return loader

    def get_meta(self, name):
        with self._engine._read():
            with self._engine._handle("r") as repo:
                try:
                    meta = dict(repo[f"{self._path}/{name}/"].attrs)
                except KeyError:
                    raise MissingMorphologyError(
                        f"`{self._engine.root}` contains no morphology named `{name}`."
                    ) from None
        meta["name"] = name
        return meta

    def has(self, name):
        with self._engine._read():
            with self._engine._handle("r") as repo:
                return f"{self._path}/{name}" in repo

    def load(self, name):
        with self._engine._read():
            with self._engine._handle("r") as repo:
                try:
                    group = repo[f"{self._path}/{name}/"]
                except:
                    raise MissingMorphologyError(
                        f"`{self._engine.root}` contains no morphology named `{name}`."
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
                            f"A morphology called '{name}' already exists in `{self._engine.root}`."
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
                    del repo[f"{self._path}/{name}"]
                except KeyError:
                    raise MorphologyRepositoryError(f"'{name}' doesn't exist.") from None

    def import_swc(self, file, name, overwrite=False):
        """
        Import and store .swc file contents as a morphology in the repository.
        """
        return self.import_arb(*arbor.load_swc_arbor(file), name, overwrite=overwrite)

    def import_asc(self, file, name, overwrite=False):
        """
        Import and store .asc file contents as a morphology in the repository.
        """
        return self.import_arb(*arbor.load_asc(file), name, overwrite=overwrite)

    def import_arb(self, morphology, labels, name, overwrite=False, centering=True):
        decor = arbor.decor()
        morpho_roots = set(
            i
            for i in range(morphology.num_branches)
            if morphology.branch_parent(i) == 4294967295
        )
        root_prox = [r[0].prox for r in map(morphology.branch_segments, morpho_roots)]
        center = np.mean([[p.x, p.y, p.z] for p in root_prox], axis=0)
        parent = None
        roots = []
        stack = []
        cable_id = morpho_roots.pop()
        while True:
            print("--- processing", cable_id)
            segments = morphology.branch_segments(cable_id)
            print(len(segments), "segments")
            if not segments:
                branch = Branch([], [], [], [])
            else:
                # Prepend the proximal end of the first segment to get [p0, p1, ..., pN]
                x = np.array([segments[0].prox.x] + [s.dist.x for s in segments])
                y = np.array([segments[0].prox.y] + [s.dist.y for s in segments])
                z = np.array([segments[0].prox.z] + [s.dist.z for s in segments])
                r = np.array(
                    [segments[0].prox.radius] + [s.dist.radius for s in segments]
                )
                if centering:
                    x -= center[0]
                    y -= center[1]
                    z -= center[2]
                branch = Branch(x, y, z, r)
            branch._cable_id = cable_id
            if parent:
                print("cable_id has parent", parent._cable_id)
                parent.attach_child(branch)
            else:
                roots.append(branch)
            children = morphology.branch_children(cable_id)
            if children:
                print("added", len(children), f"child cables ({children}) to the stack")
                stack.extend((branch, child) for child in reversed(children))
                print("next cable should be", children[0])
            if stack:
                print(len(stack), "items on stack")
                parent, cable_id = stack.pop()
                print("next parent and cable:", parent._cable_id, cable_id)
            elif not morpho_roots:
                print("out of roots, exiting")
                break
            else:
                parent = None
                cable_id = morpho_roots.pop()
                print(morphology.branch_parent(cable_id))
                print("continuing with next root", cable_id)

        morpho = Morphology(roots)
        branches = morpho.branches
        branch_map = {branch._cable_id: branch for branch in branches}
        cc = arbor.cable_cell(morphology, labels, decor)
        for label in labels:
            if "excl:" in label or label == "all":
                continue
            label_cables = cc.cables(f'"{label}"')
            print(label, label_cables)
            for cable in label_cables:
                cable_id = cable.branch
                branch = branch_map[cable_id]
                if cable.dist == 1 and cable.prox == 0:
                    branch.label_all(label)
                else:
                    prox_index = branch.get_arc_point(cable.prox, eps=1e-7)
                    print("prox arc point", prox_index)
                    if prox_index is None:
                        prox_index = branch.introduce_arc_point(cable.prox)
                    dist_index = branch.get_arc_point(cable.dist, eps=1e-7)
                    print("dist arc point", dist_index)
                    if dist_index is None:
                        dist_index = branch.introduce_arc_point(cable.dist)
                    mask = np.array(
                        [False] * prox_index
                        + [True] * (dist_index - prox_index + 1)
                        + [False] * (len(branch) - dist_index - 1)
                    )
                    branch.label_points(label, mask)

        self.save_morphology(name, morpho, overwrite=overwrite)
        return morpho


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
