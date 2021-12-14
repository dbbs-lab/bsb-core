from . import __version__
from .reporting import warn
from .helpers import ConfigurableClass, get_qualified_class_name
from .morphologies import Morphology, Compartment, Branch
from .helpers import suppress_stdout
from contextlib import contextmanager
from abc import abstractmethod, ABC
import h5py, os, time, pickle, random, numpy as np
from numpy import string_
from .exceptions import *
from .models import ConnectivitySet, PlacementSet
from sklearn.neighbors import KDTree
import os, sys, functools
import itertools as it

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "dbbs-models"))


class ResourceHandler(ABC):
    def __init__(self):
        self.handle_mode = None
        self._handle = None

    @contextmanager
    def load(self, mode="r"):
        restore_previous = False
        if mode != self.handle_mode:
            restore_previous = True
            previous_mode = self.handle_mode
            self.handle_mode = mode
            if self._handle is not None:
                self.release_handle(self._handle)
            self._handle = self.get_handle(mode)

        def handler():
            return self._handle

        try:
            yield handler  # Return the handler that always produces the current handle
        finally:
            if restore_previous:
                self.release_handle(self._handle)
                self._handle = None
                if previous_mode is not None:
                    if previous_mode == "w":
                        # Continue appending instead of re-overwriting previous write.
                        previous_mode = "a"
                    self._handle = self.get_handle(previous_mode)
                self.handle_mode = previous_mode

    @abstractmethod
    def get_handle(self, mode=None):
        """
        Open the output resource and return a handle.
        """
        pass

    @abstractmethod
    def release_handle(self, handle):
        """
        Close the open output resource and release the handle.
        """
        pass


class HDF5ResourceHandler(ResourceHandler):
    def get_handle(self, mode="r"):
        """
        Open an HDF5 resource.
        """
        # Open a new handle to the resource.
        return h5py.File(self.file, mode)

    def release_handle(self, handle):
        """
        Close the MorphologyRepository storage resource.
        """
        return handle.close()


class TreeHandler(ResourceHandler):
    """
    Interface that allows a ResourceHandler to handle storage of TreeCollections.
    """

    @abstractmethod
    def load_tree(collection_name, tree_name):
        pass

    @abstractmethod
    def store_tree_collections(self, tree_collections):
        pass

    @abstractmethod
    def list_trees(self, collection_name):
        pass


class HDF5TreeHandler(HDF5ResourceHandler, TreeHandler):
    """
    TreeHandler that uses HDF5 as resource storage
    """

    def store_tree_collections(self, tree_collections):
        with self.load("r+") as f:
            if "trees" not in f():
                tree_group = f().create_group("trees")
            else:
                tree_group = f()["trees"]
            for tree_collection in tree_collections:
                if tree_collection.name not in tree_group:
                    tree_collection_group = tree_group.create_group(tree_collection.name)
                else:
                    tree_collection_group = tree_group[tree_collection.name]
                for tree_name, tree in tree_collection.items():
                    if tree_name in tree_collection_group:
                        del tree_collection_group[tree_name]
                    tree_dataset = tree_collection_group.create_dataset(
                        tree_name, data=string_(pickle.dumps(tree))
                    )

    def load_tree(self, collection_name, tree_name):
        with self.load() as f:
            try:
                return pickle.loads(
                    f()["/trees/{}/{}".format(collection_name, tree_name)][()]
                )
            except KeyError as e:
                raise DatasetNotFoundError(
                    "Tree not found in HDF5 file '{}', path does not exist: '{}'".format(
                        f().file
                    )
                )

    def list_trees(self, collection_name):
        with self.load() as f:
            try:
                return list(f()["trees"][collection_name].keys())
            except KeyError as e:
                return DatasetNotFoundError(
                    "Tree collection '{}' not found".format(collection_name)
                )


class OutputFormatter(ConfigurableClass, TreeHandler):
    def __init__(self):
        ConfigurableClass.__init__(self)
        TreeHandler.__init__(self)
        self.save_file_as = None

    @abstractmethod
    def create_output(self):
        pass

    @abstractmethod
    def exists(self):
        """
        Check if the output file exists.
        """
        pass

    @abstractmethod
    def init_scaffold(self):
        """
        Initialize the scaffold when it has been loaded from an output file.
        """
        pass

    @abstractmethod
    def get_simulator_output_path(self, simulator_name):
        """
        Return the path where a simulator can dump preliminary output.
        """
        pass

    @abstractmethod
    def has_cells_of_type(self, name):
        """
        Check whether the position matrix for a certain cell type is present.
        """
        pass

    @abstractmethod
    def get_cells_of_type(self, name):
        """
        Return the position matrix for a specific cell type.
        """
        pass

    @abstractmethod
    def exists(self):
        """
        Check if the resource exists.
        """
        pass

    @abstractmethod
    def get_connectivity_sets(self):
        """
        Return all connectivity sets.

        :return: List of connectivity sets.
        :rtype: :class:`ConnectivitySet`
        """
        pass

    @abstractmethod
    def get_connectivity_set(self, tag):
        """
        Return a connectivity set.

        :param tag: Key of the connectivity set in the `connections` group.
        :type tag: string
        :return: The connectivity set.
        :rtype: :class:`ConnectivitySet`
        :raises: DatasetNotFoundError
        """
        pass

    @abstractmethod
    def get_connectivity_set_connection_types(self, tag):
        """
        Return the connection types that contributed to this connectivity set.
        """
        pass

    @abstractmethod
    def get_connectivity_set_meta(self, tag):
        """
        Return the meta dictionary of this connectivity set.
        """
        pass


class MorphologyRepository(HDF5TreeHandler):

    defaults = {"file": "morphology_repository.hdf5"}

    def __init__(self, file=None):
        super().__init__()
        if file is not None:
            self.file = file

    # Abstract function from ResourceHandler
    def get_handle(self, mode="r"):
        """
        Open the HDF5 storage resource and initialise the MorphologyRepository structure.
        """
        # Open a new handle to the HDF5 resource.
        handle = HDF5TreeHandler.get_handle(self, mode)
        if handle.mode != "r":
            # Repository structure missing from resource? Create it.
            self.initialise_repo_structure(handle)
        # Return the handle to the resource.
        return handle

    def initialise_repo_structure(self, handle):
        if "morphologies" not in handle:
            handle.create_group("morphologies")

    def import_swc(self, file, name, tags=[], overwrite=False):
        """
        Import and store .swc file contents as a morphology in the repository.
        """
        raise NotImplementedError("SWC temporarily unsupported.")
        # Read as CSV
        swc_data = np.loadtxt(file)

    def import_arbz(self, name, cls, overwrite=False):
        """
        Import an Arborize model as a morphology.

        Arborize models make some assumptions about morphologies, inherited from how
        NEURON deals with it: There is only 1 root, and the soma is at the beginning
        of this root. This is not necesarily so for morphologies in general in the BSB
        that can have as many roots as they want.
        """
        from patch import p

        cell = cls()
        _roots = [s for s in cell.sections if s.parent is None]
        try:
            # Offset all points to the soma point
            tx, ty, tz = cell.soma[0].x3d(0), cell.soma[0].y3d(0), cell.soma[0].z3d(0)
        except:
            # If there's no soma defined, offset to the center of the roots.
            tx = np.mean([s.x3d(0) for s in _roots])
            ty = np.mean([s.y3d(0) for s in _roots])
            tz = np.mean([s.z3d(0) for s in _roots])
        # Create a map for some important data that is only available on the Patch objects
        # and that will be lost when we use NEURON's SectionRef to retrieve connected
        # Sections and return new, stripped Section objects without this data present.
        # The map uses the unique name each section has in NEURON.
        section_id_map = {s.name(): id for id, s in enumerate(cell.sections)}
        section_labels_map = {s.name(): s.labels for id, s in enumerate(cell.sections)}
        # Keep track of which sections we visit or still have to visit to possibly warn
        # the user about unreachable/unconnected parts of the morphology.
        unvisited = set(s.name() for s in cell.sections)
        visited = set()

        # Iterate over the sections in the order that we would iterate over branches;
        # depth first from the root(s)
        def section_iter(section, parent):
            sref = p.SectionRef(section)
            yield section, parent
            for child in sref.child:
                yield from section_iter(child, section)

        # Turn any of the unsafe vector methods `x3d`, `y3d`, `z3d` or `diam3d` into
        # actual vectors.
        def vectorize(f):
            def sink():
                i = 0
                while True:
                    try:
                        yield f(i)
                    except Exception:
                        break
                    i += 1

            return np.fromiter(sink(), dtype=float)

        roots = []
        branch_map = {}
        for section, parent in it.chain(*(section_iter(s, None) for s in _roots)):
            s_name = section.name()
            try:
                unvisited.remove(s_name)
            except KeyError:
                if s_name in visited:  # pragma: nocover
                    raise CircularMorphologyError(
                        "%component% of %morphology% is visited multiple times.",
                        name,
                        s_name,
                        cell=cls.__name__,
                    ) from None
            visited.add(s_name)
            branch = Branch(
                vectorize(section.x3d) - tx,
                vectorize(section.y3d) - ty,
                vectorize(section.z3d) - tz,
                vectorize(section.diam3d) / 2.0,
            )
            branch._neuron_sid = section_id_map[s_name]
            branch.label(*section_labels_map[s_name])
            branch_map[s_name] = branch
            if parent is not None:
                branch_map[parent.name()].attach_child(branch)
            else:
                roots.append(branch)

        if unvisited:
            warn(f"{len(unvisited)} sections were not visited.", MorphologyWarning)
        strangers = set(visited) - set(s.name() for s in cell.sections)
        if strangers:
            warn(
                f"{len(strangers)} section were included that are not part of the morphology sections.",
                MorphologyWarning,
            )

        self.save_morphology(name, Morphology(roots), overwrite=overwrite)

        # Load imported morphology to test it
        morphology = self.get_morphology(name)

    def import_arbz_module(self, module):
        import arborize, inspect

        for n, c in module.__dict__.items():
            if (
                inspect.isclass(c)
                and issubclass(c, arborize.NeuronModel)
                and c != arborize.NeuronModel
            ):
                print("Importing", n)
                self.import_arbz(n, c, overwrite=True)

    def save_morphology(self, name, morphology, overwrite=False):
        with self.load("a") as repo:
            if overwrite:  # Do we overwrite previously existing dataset with same name?
                self.remove_morphology(
                    name
                )  # Delete anything that might be under this name.
            elif self.morphology_exists(name):
                raise MorphologyRepositoryError(
                    "A morphology called '{}' already exists in this repository.".format(
                        name
                    )
                )
            r = repo()["/morphologies"].create_group(name)
            b = r.create_group("branches")
            for id, branch in enumerate(morphology.branches):
                branch._tmp_id = id
                if branch._parent is not None:
                    branch._tmp_parent = branch._parent._tmp_id
                self.save_branch(name, id, branch)

    def save_branch(self, morpho_name, branch_id, branch):
        with self.load("a") as f:
            g = f()[f"/morphologies/{morpho_name}/branches"].create_group(str(branch_id))
            # Save branch parent
            if hasattr(branch, "_tmp_parent"):
                g.attrs["parent"] = branch._tmp_parent
            else:
                g.attrs["parent"] = -1
            # Save branch NEURON info
            if hasattr(branch, "_neuron_sid"):
                g.attrs["neuron_section"] = branch._neuron_sid
            # Save vectors
            for v in Branch.vectors:
                g.create_dataset(v, data=getattr(branch, v))
            # Save labels (branch labels apply to all points while the labels below are
            # vectors pertaining to single points)
            g.attrs["branch_labels"] = branch._full_labels
            l = g.create_group("labels")
            for label, label_mask in branch._label_masks.items():
                l.create_dataset(label, data=label_mask, dtype=np.bool)

    def import_repository(self, repository, overwrite=False):
        with repository.load() as external_handle:
            with self.load("a") as internal_handle:
                m_group = internal_handle()["morphologies"]
                keys = external_handle()["morphologies"].keys()
                for m_key in keys:
                    if overwrite or m_key not in m_group:
                        if m_key in m_group:
                            del m_group[m_key]
                        external_handle().copy("/morphologies/" + m_key, m_group)
                    else:
                        self.scaffold.warn(
                            "Did not import '{}' because it already existed and overwrite=False".format(
                                m_key
                            ),
                            RepositoryWarning,
                        )

    def get_morphology(self, name, scaffold=None):
        """
        Load a morphology from repository data
        """
        if name.startswith("b'"):
            name = name[2:-1]
        with self.load() as handler:
            # Check if morphology exists
            if not self.morphology_exists(name):
                raise MorphologyRepositoryError(
                    "Attempting to load unknown morphology '{}'".format(name)
                )
            group = self._raw_morphology(name, handler)
            return _morphology(group)

    def store_voxel_cloud(self, morphology, overwrite=False):
        raise NotImplementedError("Voxel cloud storage is not yet reimplemented")
        with self.load("a") as repo:
            if self.voxel_cloud_exists(morphology.morphology_name):
                if not overwrite:
                    self.scaffold.warn(
                        "Did not overwrite existing voxel cloud for '{}'".format(
                            morphology.morphology_name
                        ),
                        RepositoryWarning,
                    )
                    return
                else:
                    del repo()["/morphologies/voxel_clouds/" + morphology.morphology_name]
            voxel_cloud_group = repo()["/morphologies/voxel_clouds/"].create_group(
                morphology.morphology_name
            )
            voxel_cloud_group.attrs["name"] = morphology.morphology_name
            voxel_cloud_group.attrs["bounds"] = morphology.cloud.bounds
            voxel_cloud_group.attrs["grid_size"] = morphology.cloud.grid_size
            voxel_cloud_group.create_dataset("positions", data=morphology.cloud.voxels)
            voxel_cloud_group.create_dataset(
                "map", data=string_(pickle.dumps(morphology.cloud.map))
            )

    def morphology_exists(self, name):
        with self.load() as repo:
            return f"/morphologies/{name}" in repo()

    def voxel_cloud_exists(self, morphology_name, cloud_name):
        with self.load() as repo:
            return f"morphologies/{morphology_name}/clouds/{cloud_name}" in repo()

    def remove_morphology(self, name):
        with self.load("a") as repo:
            if self.morphology_exists(name):
                del repo()[f"/morphologies/{name}"]

    def remove_voxel_cloud(self, morphology_name, cloud_name):
        with self.load("a") as repo:
            if self.voxel_cloud_exists(name):
                del repo()[f"morphologies/{morphology_name}/clouds/{cloud_name}"]

    def list_morphologies(
        self, include_rotations=False, only_rotations=False, cell_type=None
    ):
        """
        Return a list of morphologies in a morphology repository, filtered by rotation
        and/or cell type.

        :param include_rotations: Include each cached rotation of each morphology.
        :type include_rotations: bool
        :param only_rotations: Get only the rotated caches of the morphologies.
        :type only_rotations: bool
        :param cell_type: Specify the cell type for which you want to extract the morphologies.
        :param cell_type: CellType

        :returns: List of morphology names
        :rtype: list
        """

        with self.load("r") as repo:
            # Filter out all morphology names, ignore the `voxel_clouds` category
            morpho_filter = iter(repo()["/morphologies"].keys())
            if only_rotations:
                # Exclude all non rotated names
                morpho_filter = filter(lambda x: (x.find("__") != -1), morpho_filter)
            elif not include_rotations:
                # Exclude all rotated names
                morpho_filter = filter(lambda x: (x.find("__") == -1), morpho_filter)

            # Is a cell type restriction specified?
            if cell_type is not None:
                # Filter the morphologies related to the selected cell_type
                ct_morpho = cell_type.list_all_morphologies()
                morpho_filter = filter(lambda x: x in ct_morpho, morpho_filter)
            # Apply the filter, turn it into a list.
            morphologies = list(morpho_filter)
        return morphologies

    def list_all_voxelized(self):
        with self.load() as repo:
            handle = repo()

            def morphos():
                yield from handle["/morphologies"].keys()

            def clouds(m):
                return handle[f"/morphologies/{m}/clouds"].keys()

            return [m for m in morphos() if len(clouds(m)) > 0]

    def _raw_morphology(self, name, handler):
        """
        Return the morphology data
        """
        return handler()[f"/morphologies/{name}"]

    def _raw_voxel_cloud(self, morphology_name, cloud_name, handler):
        """
        Return the voxel cloud data
        """
        return handler()[f"/morphologies/{morphology_name}/clouds/{cloud_name}"]


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
    branch.label(*attrs.get("branch_labels", iter(())))
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


class MorphologyCache:
    """
    Loads and caches :class:`morphologies <.models.Morphology>` so that each
    morphology is loaded only once and its instance is shared among all cells
    with that Morphology. Saves a lot on memory, but the Morphology should be treated as read only.
    """

    def __init__(self, morphology_repository):
        self.mr = morphology_repository

    def rotate_all_morphologies(self, phi_step, theta_step=None):
        """
        Extracts all unrotated morphologies from a morphology_repository and creates rotated versions, at sampled orientations in the 3D space

        :param phi_step: Resolution of azimuth angle sampling, in degrees
        :type phi_step: int
        :param theta_step: Resolution of elevation angle sampling, in degrees
        :type phi_step: int, optional

        """
        # Checking resolution step along the two angles - equal for both if only one value is given
        if theta_step is None:
            resolutions = [phi_step, phi_step]
        else:
            resolutions = [phi_step, theta_step]
        # Compute discretized orientations based on resolution
        phi, theta = self._discretize_orientations(*resolutions)
        # Get all unrotated morphologies from the morphology repository
        morphologies_unrotated = self.mr.list_morphologies()
        for morpho in morphologies_unrotated:
            self._construct_morphology_rotations(morpho, phi, theta)

    def rotate_morphology(self, name, phi_step, theta_step=None):
        # Checking resolution step along the two angles - equal for both if only one value is given
        if theta_step is None:
            resolutions = [phi_step, phi_step]
        else:
            resolutions = [phi_step, theta_step]
        # Compute discretized orientations based on resolution
        phi, theta = self._discretize_orientations(*resolutions)
        self._construct_morphology_rotations(name, phi, theta)

    def _discretize_orientations(self, phi_step, theta_step):
        """
        Returns two arrays of azimuth and elevation angles discretized in the 3D space
        """
        # Computing the grid of angles to discretize the 360° orientation range
        num_phi = (round(360 / phi_step) + 1) * 1j
        num_theta = (round(360 / theta_step) + 1) * 1j

        phi, theta = np.mgrid[0.0:360:num_phi, 0.0:360:num_theta]
        # From 2D to 1D arrays
        phi = phi.flatten()
        theta = theta.flatten()

        return phi, theta

    def _construct_morphology_rotations(self, morpho_name, phi, theta):
        """
        For each non existing rotation of the considered morphology morpho_name, it executes _construct_morphology_rotation
        """
        # Extract a list of rotated versions of the current morphology
        morpho_rotated_all = self.mr.list_morphologies(only_rotations=True)
        morpho_rotated = [m for m in morpho_rotated_all if m.find(morpho_name) != -1]
        # Rotating the morphology according to the discretized orientation vectors.
        for _phi, _theta in zip(map(_round, phi), map(_round, theta)):
            # Check if rotated morphology already exists
            if f"{morpho_name}__{_phi}_{_theta}" not in morpho_rotated:
                self._construct_morphology_rotation(morpho_name, _phi, _theta)

    def _construct_morphology_rotation(self, morpho_name, phi, theta):
        """
        Construct the rotated morphology according to orientation vector identified by phi_value and theta_value and save in the morphology repository
        """
        # For internal computation, angles are converted in radiants, while they are provided in degrees in function inputs or file names (more user-friendly)
        phi_rad = phi * np.pi / 180
        theta_rad = theta * np.pi / 180
        morpho = self.mr.get_morphology(morpho_name)
        start_vector = np.array([0, 1, 0])
        end_vector = np.array([np.cos(phi_rad), np.sin(phi_rad), np.sin(theta_rad)])
        morpho.rotate(start_vector, end_vector)

        self.mr.save_morphology(f"{morpho_name}__{phi}_{theta}", morpho)


_round = lambda x: int(round(x))


class HDF5Formatter(OutputFormatter, MorphologyRepository):
    """
    Stores the output of the scaffold as a single HDF5 file. Is also a MorphologyRepository
    and an HDF5TreeHandler.
    """

    defaults = {
        "file": "scaffold_network_{}.hdf5".format(
            time.strftime("%Y_%m_%d-%H%M%S") + str(random.random()).split(".")[1]
        ),
        "simulator_output_path": False,
        "morphology_repository": None,
    }

    def create_output(self):
        was_compiled = self.exists()
        if was_compiled:
            with h5py.File("__backup__.hdf5", "w") as backup:
                with self.load() as repo:
                    repo().copy("/morphologies", backup)

        if self.save_file_as:
            self.file = self.save_file_as

        try:
            with self.load("w") as output:
                self.store_configuration()
                self.store_cells()
                self.store_tree_collections(self.scaffold.trees.__dict__.values())
                self.store_statistics()
                self.store_appendices()
                self.store_morphology_repository(was_compiled)
        except:
            os.remove(self.file)
            raise

        if was_compiled:
            os.remove("__backup__.hdf5")

    def exists(self):
        return os.path.exists(self.file)

    def init_scaffold(self):
        scf = self.scaffold
        with self.load() as res:
            for cell_type_name, count in res()["statistics/cells_placed"].attrs.items():
                scf.statistics.cells_placed[cell_type_name] = count
            for tag in res()["cells/connections"]:
                dataset = res()["cells/connections/" + tag]
                for contributing_type in dataset.attrs["connection_types"]:
                    scf.configuration.connection_types[contributing_type].tags.append(tag)
            scf.labels = {l: v[()] for l, v in res()["cells/labels"].items()}
            sets = (v["identifiers"] for v in res()["cells/placement"].values())
            max_ids = (id[::2] + id[1::2] if len(id) else [0] for id in sets)
            scf._nextId = functools.reduce(max, map(np.max, max_ids), 0)

    def validate(self):
        pass

    def store_configuration(self, config=None):
        config = config if config is not None else self.scaffold.configuration
        with self.load("a") as f:
            f = f()
            f.attrs["version"] = __version__
            f.attrs["configuration_name"] = config._name
            f.attrs["configuration_type"] = config._type
            f.attrs["configuration_class"] = get_qualified_class_name(config)
            # REALLY BAD HACK: This is to cover up #222 in the test networks during unit testing.
            f.attrs["configuration_string"] = config._raw.replace(
                '"simulation_volume_x": 400.0', '"simulation_volume_x": ' + str(config.X)
            ).replace(
                '"simulation_volume_z": 400.0', '"simulation_volume_z": ' + str(config.Z)
            )

    def store_cells(self):
        with self.load("a") as f:
            cells_group = f().require_group("cells")
            self.store_placement(cells_group)
            self.store_cell_connections(cells_group)
            self.store_labels(cells_group)

    def store_placement(self, cells_group):
        placement = cells_group.require_group("placement")
        for cell_type in self.scaffold.get_cell_types():
            cell_type_group = placement.create_group(cell_type.name)
            ids = cell_type_group.create_dataset(
                "identifiers", data=cell_type._ser_cached_ids(), dtype=np.int32
            )
            if not cell_type.entity:
                cell_type_group.create_dataset(
                    "positions", data=self.scaffold.cells_by_type[cell_type.name][:, 2:5]
                )
            if cell_type.name in self.scaffold.rotations.keys():
                cell_type_group.create_dataset(
                    "rotations", data=self.scaffold.rotations[cell_type.name]
                )

    def store_cell_connections(self, cells_group):
        scf = self.scaffold
        connections_group = cells_group.require_group("connections")
        compartments_group = cells_group.require_group("connection_compartments")
        morphologies_group = cells_group.require_group("connection_morphologies")
        for tag, connectome_data in scf.cell_connections_by_tag.items():
            _map = f"__map_{tag}"
            related_types = [
                conn_t
                for conn_t in scf.configuration.connection_types.values()
                if tag in conn_t.tags
            ]
            connection_dataset = connections_group.create_dataset(
                tag, data=connectome_data
            )
            connection_dataset.attrs["tag"] = tag
            connection_dataset.attrs["connection_types"] = [t.name for t in related_types]
            connection_dataset.attrs["connection_type_classes"] = list(
                map(get_qualified_class_name, related_types)
            )
            if tag in scf._connectivity_set_meta:
                meta_dict = scf._connectivity_set_meta[tag]
                for key in meta_dict:
                    connection_dataset.attrs[key] = meta_dict[key]
            if tag in scf.connection_compartments:
                compartments_group.create_dataset(
                    tag, data=scf.connection_compartments[tag], dtype=int
                )
                morphology_dataset = morphologies_group.create_dataset(
                    tag, data=scf.connection_morphologies[tag], dtype=int
                )
                # Sanitize values to pure Python strings. H5py errors on numpy str
                safe_map = [str(x) for x in scf.connection_morphologies[_map]]
                morphology_dataset.attrs["map"] = safe_map

    def store_labels(self, cells_group):
        labels_group = cells_group.create_group("labels")
        for label, data in self.scaffold.labels.items():
            # Make sure the data is one-dimensional
            vector = np.array(data).reshape(-1)
            labels_group.create_dataset(label, data=vector)

    def store_statistics(self):
        with self.load("a") as f:
            statistics = f().create_group("statistics")
            self.store_placement_statistics(statistics)

    def store_placement_statistics(self, statistics_group):
        storage_group = statistics_group.create_group("cells_placed")
        for key, value in self.scaffold.statistics.cells_placed.items():
            storage_group.attrs[key] = value

    def store_appendices(self):
        # Append extra datasets specified internally or by user.
        with self.load("a") as f:
            for key, data in self.scaffold.appends.items():
                dset = f().create_dataset(key, data=data)

    def load_appendix(self, key):
        with self.load("r") as f:
            return f()[key][()]

    def store_morphology_repository(self, was_compiled=False):
        with self.load("a") as resource:
            if was_compiled:  # File already existed?
                # Copy from the backup of previous version
                with h5py.File("__backup__.hdf5", "r") as backup:
                    if "morphologies" in resource():
                        del resource()["/morphologies"]
                    backup.copy("/morphologies", resource())
            else:  # Fresh compilation
                self.initialise_repo_structure(resource())
                if self.morphology_repository is not None:  # Repo specified
                    self.import_repository(self.scaffold.morphology_repository)

    def get_simulator_output_path(self, simulator_name):
        return self.simulator_output_path or os.getcwd()

    def has_cells_of_type(self, name, entity=False):
        with self.load() as resource:
            return name in resource()["/cells/placement"]

    def get_cells_of_type(self, name, entity=False):
        # Check if cell type is present
        if not self.has_cells_of_type(name, entity=entity):
            raise DatasetNotFoundError(
                "Attempting to load {} type '{}' that isn't defined in the storage.".format(
                    "cell" if not entity else "entity", name
                )
            )
        with self.load() as resource:
            ps = self.scaffold.get_cell_type(name).get_placement_set()
            ids = ps.identifiers
            if entity:
                spoof_matrix = ids
            else:
                spoof_matrix = np.column_stack(
                    (
                        ids,
                        np.zeros(len(ids)),
                        ps.positions,
                    )
                )
            return spoof_matrix

    def get_connectivity_set_connection_types(self, tag):
        """
        Return all the ConnectionStrategies that contributed to the creation of this
        connectivity set.
        """
        with self.load() as f:
            # Get list of contributing types
            type_list = f()["cells/connections/" + tag].attrs["connection_types"]
            # Map contributing type names to contributing types
            return list(
                map(lambda name: self.scaffold.get_connection_type(name), type_list)
            )

    def get_connectivity_set_meta(self, tag):
        """
        Return the metadata associated with this connectivity set.
        """
        with self.load() as f:
            return dict(f()["cells/connections/" + tag].attrs)

    def get_connectivity_sets(self):
        """
        Return all the ConnectivitySets present in the network file.
        """
        with self.load() as f:
            return list(
                ConnectivitySet(self, tag) for tag in f()["cells/connections/"].keys()
            )

    def get_connectivity_set(self, tag):
        return ConnectivitySet(self, tag)

    def get_placement_set(self, type):
        return PlacementSet(self, type)

    @classmethod
    def reconfigure(cls, hdf5_file, config):
        if not os.path.exists(hdf5_file):
            raise FileNotFoundError(
                "HDF5 file '{}' to reconfigure does not exist.".format(hdf5_file)
            )
        hdf5_formatter = cls()
        hdf5_formatter.file = hdf5_file
        hdf5_formatter.store_configuration(config)
