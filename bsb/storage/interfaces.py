import abc
import types
import functools
from contextlib import contextmanager
import numpy as np
import arbor
from ..morphologies import Morphology, Branch
from ..trees import BoxTree
from rtree import index as rtree
from scipy.spatial.transform import Rotation


class Interface(abc.ABC):
    _iface_engine_key = None

    def __init__(self, handler):
        self._handler = handler

    def __init_subclass__(cls, **kwargs):
        # Only change engine key if explicitly given.
        if "engine_key" in kwargs:
            cls._iface_engine_key = kwargs["engine_key"]


class Engine(Interface):
    def __init__(self, root):
        self.root = root

    @property
    def format(self):
        # This attribute is set on the engine by the storage provider and correlates to
        # the name of the engine plugin.
        return self._format

    @abc.abstractmethod
    def exists(self):
        pass

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def move(self, new_root):
        pass

    @abc.abstractmethod
    def remove(self):
        pass


class NetworkDescription(Interface):
    pass


class FileStore(Interface, engine_key="files"):
    @abc.abstractmethod
    def store(self, content, id=None, meta=None):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def remove(self, id):
        pass

    @abc.abstractmethod
    def store_active_config(self, config):
        pass

    @abc.abstractmethod
    def load_active_config(self):
        pass


class PlacementSet(Interface):
    @abc.abstractmethod
    def __init__(self, engine, type):
        pass

    @abc.abstractclassmethod
    def create(cls, engine, type):
        """
        Override with a method to create the placement set.
        """
        pass

    @abc.abstractstaticmethod
    def exists(self, engine, type):
        """
        Override with a method to check existence of the placement set
        """
        pass

    def require(self, engine, type):
        """
        Can be overridden with a method to make sure the placement set exists. The default
        implementation uses the class's ``exists`` and ``create`` methods.
        """
        if not self.exists(engine, type):
            self.create(engine, type)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Override with a method to clear (some chunks of) the placement set
        """
        pass

    @abc.abstractmethod
    def get_all_chunks(self):
        pass

    @abc.abstractmethod
    def load_positions(self):
        """
        Return a dataset of cell positions.
        """
        pass

    @abc.abstractmethod
    def load_rotations(self):
        """
        Return a dataset of cell rotations.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        pass

    @abc.abstractmethod
    def load_morphologies(self):
        """
        Return a :class:`~.storage.interfaces.MorphologySet` associated to the cells.

        :return: Set of morphologies
        :rtype: :class:`~.storage.interfaces.MorphologySet`

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def append_data(
        self, chunk, positions=None, morphologies=None, rotations=None, additional=None
    ):
        pass

    @abc.abstractmethod
    def append_additional(self, name, chunk, data):
        pass

    def load_boxes(self, cache=None, itr=True):
        print("boxes of", self.type.name)
        if cache is None:
            print("no cache")
            mset = self.load_morphologies()
            print(len(mset), "morphologies stored")
        else:
            mset = cache
            print(len(mset), "morphologies cached")
        expansion = [*zip([0] * 4 + [1] * 4, ([0] * 2 + [1] * 2) * 2, [0, 1] * 4)]
        print("hihi expansion table", expansion)

        def _box_of(m, o, r):
            oo = (m["ldc"], m["mdc"])
            # Make the 8 corners of the box
            corners = np.array([[oo[x][0], oo[y][1], oo[z][2]] for x, y, z in expansion])
            # Rotate them
            rotbox = Rotation.from_euler("xyz", r).apply(corners)
            # Find outer box of rotated and translated starting box
            return np.concatenate(
                (np.min(rotbox, axis=0) + o, np.max(rotbox, axis=0) + o)
            )

        iters = (mset.iter_meta(), self.load_positions(), self.load_rotations())
        iter = map(_box_of, *iters)
        if itr:
            return iter
        else:
            return list(iter)

    def load_box_tree(self, cache=None):
        return BoxTree(self.load_boxes(cache=cache, itr=True))


class MorphologyRepository(Interface, engine_key="morphologies"):
    @abc.abstractmethod
    def select(self, selector):
        pass

    @abc.abstractmethod
    def save(self, selector):
        pass

    @abc.abstractmethod
    def has(self, selector):
        pass

    @abc.abstractmethod
    def preload(self, selector):
        pass

    @abc.abstractmethod
    def load(self, selector):
        pass

    def import_swc(self, file, name, overwrite=False):
        """
        Import and store .swc file contents as a morphology in the repository.
        """
        labels = arbor.label_dict(
            dict(soma="(tag 1)", axon="(tag 2)", dendrites="(tag 3)")
        )
        try:
            morpho = arbor.load_swc_arbor(file)
        except RuntimeError as e:
            warn(f"Couldn't parse swc: `{e}`. Falling back to NEURON-style parsing.")
            morpho = arbor.load_swc_neuron(file)

        return self.import_arb(morpho, labels, name, overwrite=overwrite)

    def import_asc(self, file, name, overwrite=False):
        """
        Import and store .asc file contents as a morphology in the repository.
        """
        asc = arbor.load_asc(file)
        return self.import_arb(asc.morpho, asc.labels, name, overwrite=overwrite)

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
            segments = morphology.branch_segments(cable_id)
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
                parent.attach_child(branch)
            else:
                roots.append(branch)
            children = morphology.branch_children(cable_id)
            if children:
                stack.extend((branch, child) for child in reversed(children))
            if stack:
                parent, cable_id = stack.pop()
            elif not morpho_roots:
                break
            else:
                parent = None
                cable_id = morpho_roots.pop()

        morpho = Morphology(roots)
        branches = morpho.branches
        branch_map = {branch._cable_id: branch for branch in branches}
        cc = arbor.cable_cell(morphology, labels, decor)
        for label in labels:
            if "excl:" in label or label == "all":
                continue
            label_cables = cc.cables(f'"{label}"')
            for cable in label_cables:
                cable_id = cable.branch
                branch = branch_map[cable_id]
                if cable.dist == 1 and cable.prox == 0:
                    branch.label_all(label)
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
                    branch.label_points(label, mask)

        self.save(name, morpho, overwrite=overwrite)
        return morpho


class ConnectivitySet(Interface):
    @abc.abstractclassmethod
    def create(cls, engine, tag):
        """
        Override with a method to create the placement set.
        """
        pass

    @abc.abstractstaticmethod
    def exists(self, engine, tag):
        """
        Override with a method to check existence of the placement set
        """
        pass

    def require(self, engine, tag):
        """
        Can be overridden with a method to make sure the placement set exists. The default
        implementation uses the class's ``exists`` and ``create`` methods.
        """
        if not self.exists(engine, tag):
            self.create(engine, tag)

    @abc.abstractmethod
    def clear(self, chunks=None):
        """
        Override with a method to clear (some chunks of) the placement set
        """
        pass

    @abc.abstractclassmethod
    def get_tags(cls, engine):
        pass


class Label(Interface):
    @abc.abstractmethod
    def label(self, identifiers):
        pass

    @abc.abstractmethod
    def unlabel(self, identifiers):
        pass

    @abc.abstractmethod
    def store(self, identifiers):
        pass

    @property
    @abc.abstractmethod
    def cells(self):
        pass

    @abc.abstractmethod
    def list(self):
        pass


class StoredMorphology:
    def __init__(self, name, loader, meta):
        self.name = name
        self._loader = loader
        self._meta = meta

    def get_meta(self):
        return self._meta.copy()

    def load(self):
        return self._loader()

    @functools.cache
    def cached_load(self):
        return self.load()
