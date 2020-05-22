from ...interfaces import Engine
from contextlib import contextmanager
import h5py, os
from .placement_set import PlacementSet
from .configuration import Configuration
from .label import Label
from .filter import Filter


class HDF5Engine(Engine):
    def __init__(self, root):
        super().__init__(root)
        self.handle_mode = None
        self._handle = None
        self.file = root

    def get_handle(self, mode="r"):
        """
            Open an HDF5 file.
        """
        # Open a new handle to the resource.
        return h5py.File(self.file, mode)

    def release_handle(self, handle):
        """
            Close the HDF5 file.
        """
        return handle.close()

    @contextmanager
    def open(self, mode="r"):
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

    def exists(self):
        return os.path.exists(self.file)

    def create(self):
        with self.open("w") as handler:
            handle = handler()
            handle.create_group("cells")
            handle.create_group("cells/placement")
            handle.create_group("cells/connections")
            handle.create_group("cells/labels")

    def move(self, new_root):
        from shutil import copy2

        copy2(self.file, new_root)
        self.file = new_root

    def remove(self):
        os.remove(self.file)


#
#
#
#
# class HDF5TreeHandler(HDF5Engine, TreeHandler):
#     """
#         TreeHandler that uses HDF5 as resource storage
#     """
#
#     def store_tree_collections(self, tree_collections):
#         with self.open("r+") as f:
#             if "trees" not in f():
#                 tree_group = f().create_group("trees")
#             else:
#                 tree_group = f()["trees"]
#             for tree_collection in tree_collections:
#                 if tree_collection.name not in tree_group:
#                     tree_collection_group = tree_group.create_group(tree_collection.name)
#                 else:
#                     tree_collection_group = tree_group[tree_collection.name]
#                 for tree_name, tree in tree_collection.items():
#                     if tree_name in tree_collection_group:
#                         del tree_collection_group[tree_name]
#                     tree_dataset = tree_collection_group.create_dataset(
#                         tree_name, data=string_(pickle.dumps(tree))
#                     )
#
#     def load_tree(self, collection_name, tree_name):
#         with self.open() as f:
#             try:
#                 return pickle.loads(
#                     f()["/trees/{}/{}".format(collection_name, tree_name)][()]
#                 )
#             except KeyError as e:
#                 raise DatasetNotFoundError(
#                     "Tree not found in HDF5 file '{}', path does not exist: '{}'".format(
#                         f().file
#                     )
#                 )
#
#     def list_trees(self, collection_name):
#         with self.open() as f:
#             try:
#                 return list(f()["trees"][collection_name].keys())
#             except KeyError as e:
#                 return DatasetNotFoundError(
#                     "Tree collection '{}' not found".format(collection_name)
#                 )
#
#
# class MorphologyRepository(HDF5TreeHandler):
#
#     defaults = {"file": "morphology_repository.hdf5"}
#
#     protected_keys = ["voxel_clouds"]
#
#     def __init__(self, file=None):
#         super().__init__()
#         if file is not None:
#             self.file = file
#
#     # Abstract function from Engine
#     def get_handle(self, mode="r"):
#         """
#             Open the HDF5 storage resource and initialise the MorphologyRepository structure.
#         """
#         # Open a new handle to the HDF5 resource.
#         handle = HDF5TreeHandler.get_handle(self, mode)
#         if handle.mode != "r":
#             # Repository structure missing from resource? Create it.
#             self.initialise_repo_structure(handle)
#         # Return the handle to the resource.
#         return handle
#
#     def initialise_repo_structure(self, handle):
#         if "morphologies" not in handle:
#             handle.create_group("morphologies")
#         if "morphologies/voxel_clouds" not in handle:
#             handle.create_group("morphologies/voxel_clouds")
#
#     def import_swc(self, file, name, tags=[], overwrite=False):
#         """
#             Import and store .swc file contents as a morphology in the repository.
#         """
#         # Read as CSV
#         swc_data = np.loadtxt(file)
#         # Create empty dataset
#         dataset_length = len(swc_data)
#         dataset_data = np.empty((dataset_length, 10))
#         # Map parent id's to start coordinates. Root node (id: -1) is at 0., 0., 0.
#         starts = {-1: [0.0, 0.0, 0.0]}
#         id_map = {-1: -1}
#         next_id = 1
#         # Get translation for a new space with compartment 0 as origin.
#         translation = swc_data[0, 2:5]
#         # Iterate over the compartments
#         for i in range(dataset_length):
#             # Extract compartment record
#             compartment = swc_data[i, :]
#             # Renumber the compartments to yield a continuous incrementing list of IDs
#             # (increases performance of graph theory and network related tasks)
#             compartment_old_id = compartment[0]
#             compartment_id = next_id
#             next_id += 1
#             # Keep track of a map to translate old IDs to new IDs
#             id_map[compartment_old_id] = compartment_id
#             compartment_type = compartment[1]
#             # Check if parent id is known
#             if not compartment[6] in id_map:
#                 raise MorphologyDataError(
#                     "Node {} references a parent node {} that isn't known yet".format(
#                         compartment_old_id, compartment[6]
#                     )
#                 )
#             # Map the old parent ID to the new parent ID
#             compartment_parent = id_map[compartment[6]]
#             # Use parent endpoint as startpoint, get endpoint and store it as a startpoint for child compartments
#             compartment_start = starts[compartment_parent]
#             # Translate each compartment to a new space with compartment 0 as origin.
#             compartment_end = compartment[2:5] - translation
#             starts[compartment_id] = compartment_end
#             # Get more compartment radius
#             compartment_radius = compartment[5]
#             # Store compartment in the repository dataset
#             dataset_data[i] = [
#                 compartment_id,
#                 compartment_type,
#                 *compartment_start,
#                 *compartment_end,
#                 compartment_radius,
#                 compartment_parent,
#             ]
#         # Save the dataset in the repository
#         self.save_morphology_dataset(name, dataset_data, overwrite=overwrite)
#
#     def import_arbz(self, name, cls, overwrite=False):
#         from neuron import h
#
#         cell = cls()
#         c_types = Morphology.compartment_types
#         # Initialization
#         idx = 0
#         orphans = []
#         section_to_id = dict()
#         compartments = []
#         dataset = []
#         # Translate all points so that soma[0] = 0., 0., 0.
#         tx = cell.sections[0].x3d(0)
#         ty = cell.sections[0].y3d(0)
#         tz = cell.sections[0].z3d(0)
#         # Loop over sections to extract compartments
#         for id, section in enumerate(cell.sections):
#             section_to_id[section.name()] = id
#             # Keep the highest ctype value to retain the most specific section label
#             label = np.max([c_types[l] for l in section.labels])
#             for p in range(section.n3d() - 1):
#                 data = [
#                     idx + p,
#                     label,
#                     section.x3d(p) - tx,
#                     section.y3d(p) - ty,
#                     section.z3d(p) - tz,
#                     section.x3d(p + 1) - tx,
#                     section.y3d(p + 1) - ty,
#                     section.z3d(p + 1) - tz,
#                     section.diam3d(p) / 2.0,
#                     idx + p - 1,
#                     id,
#                 ]
#                 c = Compartment(None, data)
#                 if p == 0:
#                     c.parent = -1
#                     orphans.append(c)
#                 compartments.append(c)
#                 dataset.append(data)
#             idx += p + 1
#         # Fix orphans
#         for orphan in orphans:
#             # Tricky NEURON workaround to get the parent section id.
#             section = cell.sections[orphan.section_id]
#             sec_ref = h.SectionRef(sec=section.__neuron__())
#             try:
#                 with suppress_stdout():
#                     parent_section = sec_ref.parent().sec
#             except Exception as e:
#                 continue
#             try:
#                 parent_section_id = section_to_id[parent_section.name()]
#             except KeyError:
#                 raise MorphologyDataError(
#                     "Arborize model {} connects section '{}' to '{}' which is not part of the morphology.".format(
#                         name, section.name(), parent_section.name()
#                     )
#                     + " In order to be a part of the morphology, the section needs to occur in `self.sections`"
#                 )
#             # Get the id of the last compartment of the parent section.
#             last_compartment = list(
#                 filter(lambda c: c.section_id == parent_section_id, compartments)
#             )[-1]
#             # Overwrite the parent id column of the orphan.
#             dataset[orphan.id][9] = last_compartment.id
#         self.save_morphology_dataset(name, dataset, overwrite=overwrite)
#
#         # Load imported morphology to test it
#         morphology = self.get_morphology(name)
#
#     def import_arbz_module(self, module):
#         import arborize, inspect
#
#         for n, c in module.__dict__.items():
#             if (
#                 inspect.isclass(c)
#                 and issubclass(c, arborize.NeuronModel)
#                 and c != arborize.NeuronModel
#             ):
#                 print("Importing", n)
#                 self.import_arbz(n, c, overwrite=True)
#
#     def save_morphology_dataset(self, name, data, overwrite=False):
#         with self.open("a") as repo:
#             if overwrite:  # Do we overwrite previously existing dataset with same name?
#                 self.remove_morphology(
#                     name
#                 )  # Delete anything that might be under this name.
#             elif self.morphology_exists(name):
#                 raise MorphologyRepositoryError(
#                     "A morphology called '{}' already exists in this repository.".format(
#                         name
#                     )
#                 )
#             data = np.array(data)
#             # Create the dataset
#             dataset = repo()["morphologies"].create_dataset(name, data=data)
#             # Set attributes
#             dataset.attrs["name"] = name
#             dataset.attrs["search_radii"] = np.max(np.abs(data[:, 2:5]), axis=0)
#             dataset.attrs["type"] = "swc"
#
#     def import_repository(self, repository, overwrite=False):
#         with repository.open() as external_handle:
#             with self.open("a") as internal_handle:
#                 m_group = internal_handle()["morphologies"]
#                 keys = external_handle()["morphologies"].keys()
#                 for m_key in keys:
#                     if m_key not in self.protected_keys:
#                         if overwrite or m_key not in m_group:
#                             if m_key in m_group:
#                                 del m_group[m_key]
#                             external_handle().copy("/morphologies/" + m_key, m_group)
#                         else:
#                             self.scaffold.warn(
#                                 "Did not import '{}' because it already existed and overwrite=False".format(
#                                     m_key
#                                 ),
#                                 RepositoryWarning,
#                             )
#
#     def get_morphology(self, name, scaffold=None):
#         """
#             Load a morphology from repository data
#         """
#         with self.open() as handler:
#             # Check if morphology exists
#             if not self.morphology_exists(name):
#                 raise MorphologyRepositoryError(
#                     "Attempting to load unknown morphology '{}'".format(name)
#                 )
#             # Take out all the data with () index, and send along the metadata stored in the attributes
#             data = self._raw_morphology(name, handler)
#             repo_data = data[()]
#             repo_meta = dict(data.attrs)
#             voxel_kwargs = {}
#             if self.voxel_cloud_exists(name):
#                 voxels = self._raw_voxel_cloud(name, handler)
#                 voxel_kwargs["voxel_data"] = voxels["positions"][()]
#                 voxel_kwargs["voxel_meta"] = dict(voxels.attrs)
#                 voxel_kwargs["voxel_map"] = pickle.loads(voxels["map"][()])
#             return Morphology.from_repo_data(
#                 repo_data, repo_meta, scaffold=scaffold, **voxel_kwargs
#             )
#
#     def store_voxel_cloud(self, morphology, overwrite=False):
#         with self.open("a") as repo:
#             if self.voxel_cloud_exists(morphology.morphology_name):
#                 if not overwrite:
#                     self.scaffold.warn(
#                         "Did not overwrite existing voxel cloud for '{}'".format(
#                             morphology.morphology_name
#                         ),
#                         RepositoryWarning,
#                     )
#                     return
#                 else:
#                     del repo()["/morphologies/voxel_clouds/" + morphology.morphology_name]
#             voxel_cloud_group = repo()["/morphologies/voxel_clouds/"].create_group(
#                 morphology.morphology_name
#             )
#             voxel_cloud_group.attrs["name"] = morphology.morphology_name
#             voxel_cloud_group.attrs["bounds"] = morphology.cloud.bounds
#             voxel_cloud_group.attrs["grid_size"] = morphology.cloud.grid_size
#             voxel_cloud_group.create_dataset("positions", data=morphology.cloud.voxels)
#             voxel_cloud_group.create_dataset(
#                 "map", data=string_(pickle.dumps(morphology.cloud.map))
#             )
#
#     def morphology_exists(self, name):
#         with self.open() as repo:
#             return name in repo()["morphologies"]
#
#     def voxel_cloud_exists(self, name):
#         with self.open() as repo:
#             return name in repo()["morphologies/voxel_clouds"]
#
#     def remove_morphology(self, name):
#         with self.open("a") as repo:
#             if self.morphology_exists(name):
#                 del repo()["morphologies/" + name]
#
#     def remove_voxel_cloud(self, name):
#         with self.open("a") as repo:
#             if self.voxel_cloud_exists(name):
#                 del repo()["morphologies/voxel_clouds/" + name]
#
#     def list_all_morphologies(self):
#         with self.open() as repo:
#             return list(
#                 filter(lambda x: x != "voxel_clouds", repo()["morphologies"].keys())
#             )
#
#     def list_all_voxelized(self):
#         with self.open() as repo:
#             all = list(repo()["morphologies"].keys())
#             voxelized = list(
#                 filter(lambda x: x in repo()["/morphologies/voxel_clouds"], all)
#             )
#             return voxelized
#
#     def _raw_morphology(self, name, handler):
#         """
#             Return the morphology dataset
#         """
#         return handler()["morphologies/" + name]
#
#     def _raw_voxel_cloud(self, name, handler):
#         """
#             Return the morphology dataset
#         """
#         return handler()["morphologies/voxel_clouds/" + name]
#
#
# class HDF5Formatter(Storage, MorphologyRepository):
#     """
#         Stores the output of the scaffold as a single HDF5 file. Is also a MorphologyRepository
#         and an HDF5TreeHandler.
#     """
#
#     defaults = {
#         "file": "scaffold_network_{}.hdf5".format(time.strftime("%Y_%m_%d-%H%M%S")),
#         "simulator_output_path": False,
#         "morphology_repository": None,
#     }
#
#     def create_output(self):
#         was_compiled = self.exists()
#         if was_compiled:
#             with h5py.File("__backup__.hdf5", "w") as backup:
#                 with self.open() as repo:
#                     repo().copy("/morphologies", backup)
#
#         if self.save_file_as:
#             self.file = self.save_file_as
#
#         with self.open("w") as output:
#             self.store_configuration()
#             self.store_cells()
#             self.store_entities()
#             self.store_tree_collections(self.scaffold.trees.__dict__.values())
#             self.store_statistics()
#             self.store_appendices()
#             self.store_morphology_repository(was_compiled)
#
#         if was_compiled:
#             os.remove("__backup__.hdf5")
#
#     def exists(self):
#         return os.path.exists(self.file)
#
#     def init_scaffold(self):
#         with self.open() as resource:
#             self.scaffold.configuration.cell_type_map = resource()["cells"].attrs["types"]
#             self.scaffold.placement_stitching = resource()["cells/stitching"][:]
#             for cell_type_name, count in resource()[
#                 "statistics/cells_placed"
#             ].attrs.items():
#                 self.scaffold.statistics.cells_placed[cell_type_name] = count
#             for tag in resource()["cells/connections"]:
#                 dataset = resource()["cells/connections/" + tag]
#                 for contributing_type in dataset.attrs["connection_types"]:
#                     self.scaffold.configuration.connection_types[
#                         contributing_type
#                     ].tags.append(tag)
#             self.scaffold.labels = {
#                 l: resource()["cells/labels/" + l][()] for l in resource()["cells/labels"]
#             }
#
#     def validate(self):
#         pass
#
#     def store_configuration(self, config=None):
#         config = config if config is not None else self.scaffold.configuration
#         with self.open("a") as f:
#             f = f()
#             f.attrs["version"] = __version__
#             f.attrs["configuration_name"] = config._name
#             f.attrs["configuration_type"] = config._type
#             f.attrs["configuration_class"] = get_qualified_class_name(config)
#             # REALLY BAD HACK: This is to cover up #222 in the test networks during unit testing.
#             f.attrs["configuration_string"] = config._raw.replace(
#                 '"simulation_volume_x": 400.0', '"simulation_volume_x": ' + str(config.X)
#             ).replace(
#                 '"simulation_volume_z": 400.0', '"simulation_volume_z": ' + str(config.Z)
#             )
#
#     def store_cells(self):
#         with self.open("a") as f:
#             cells_group = f().create_group("cells")
#             cells_group.create_dataset(
#                 "stitching", data=self.scaffold.placement_stitching
#             )
#             self.store_cell_positions(cells_group)
#             self.store_placement(cells_group)
#             self.store_cell_connections(cells_group)
#             self.store_labels(cells_group)
#
#     def store_entities(self):
#         with self.open("a") as f:
#             cells_group = f().create_group("entities")
#             for key, data in self.scaffold.entities_by_type.items():
#                 cells_group.create_dataset(key, data=data)
#
#     def store_placement(self, cells_group):
#         placement = cells_group.create_group("placement")
#         for cell_type in self.scaffold.get_cell_types():
#             cell_type_group = placement.create_group(cell_type.name)
#             ids = cell_type_group.create_dataset(
#                 "identifiers", data=cell_type.serialize_identifiers(), dtype=np.int32
#             )
#             if not cell_type.entity:
#                 cell_type_group.create_dataset(
#                     "positions", data=self.scaffold.cells_by_type[cell_type.name][:, 2:5]
#                 )
#
#     def store_cell_positions(self, cells_group):
#         position_dataset = cells_group.create_dataset(
#             "positions", data=self.scaffold.cells
#         )
#         cell_type_names = self.scaffold.configuration.cell_type_map
#         cells_group.attrs["types"] = cell_type_names
#         type_maps_group = cells_group.create_group("type_maps")
#         for type in self.scaffold.configuration.cell_types.keys():
#             type_maps_group.create_dataset(
#                 type + "_map",
#                 data=np.where(self.scaffold.cells[:, 1] == cell_type_names.index(type))[
#                     0
#                 ],
#             )
#
#     def store_cell_connections(self, cells_group):
#         connections_group = cells_group.create_group("connections")
#         compartments_group = cells_group.create_group("connection_compartments")
#         morphologies_group = cells_group.create_group("connection_morphologies")
#         for tag, connectome_data in self.scaffold.cell_connections_by_tag.items():
#             related_types = list(
#                 filter(
#                     lambda x: tag in x.tags,
#                     self.scaffold.configuration.connection_types.values(),
#                 )
#             )
#             connection_dataset = connections_group.create_dataset(
#                 tag, data=connectome_data
#             )
#             connection_dataset.attrs["tag"] = tag
#             connection_dataset.attrs["connection_types"] = list(
#                 map(lambda x: x.name, related_types)
#             )
#             connection_dataset.attrs["connection_type_classes"] = list(
#                 map(get_qualified_class_name, related_types)
#             )
#             if tag in self.scaffold._connectivity_set_meta:
#                 meta_dict = self.scaffold._connectivity_set_meta[tag]
#                 for key in meta_dict:
#                     connection_dataset.attrs[key] = meta_dict[key]
#             if tag in self.scaffold.connection_compartments:
#                 compartments_group.create_dataset(
#                     tag, data=self.scaffold.connection_compartments[tag], dtype=int
#                 )
#                 morphology_dataset = morphologies_group.create_dataset(
#                     tag, data=self.scaffold.connection_morphologies[tag], dtype=int
#                 )
#                 morphology_dataset.attrs["map"] = self.scaffold.connection_morphologies[
#                     tag + "_map"
#                 ]
#
#     def store_labels(self, cells_group):
#         labels_group = cells_group.create_group("labels")
#         for label in self.scaffold.labels.keys():
#             labels_group.create_dataset(label, data=self.scaffold.labels[label])
#
#     def store_statistics(self):
#         with self.open("a") as f:
#             statistics = f().create_group("statistics")
#             self.store_placement_statistics(statistics)
#
#     def store_placement_statistics(self, statistics_group):
#         storage_group = statistics_group.create_group("cells_placed")
#         for key, value in self.scaffold.statistics.cells_placed.items():
#             storage_group.attrs[key] = value
#
#     def store_appendices(self):
#         # Append extra datasets specified internally or by user.
#         with self.open("a") as f:
#             for key, data in self.scaffold.appends.items():
#                 dset = f().create_dataset(key, data=data)
#
#     def store_morphology_repository(self, was_compiled=False):
#         with self.open("a") as resource:
#             if was_compiled:  # File already existed?
#                 # Copy from the backup of previous version
#                 with h5py.File("__backup__.hdf5", "r") as backup:
#                     if "morphologies" in resource():
#                         del resource()["/morphologies"]
#                     backup.copy("/morphologies", resource())
#             else:  # Fresh compilation
#                 self.initialise_repo_structure(resource())
#                 if self.morphology_repository is not None:  # Repo specified
#                     self.import_repository(self.scaffold.morphology_repository)
#
#     def get_simulator_output_path(self, simulator_name):
#         return self.simulator_output_path or os.getcwd()
#
#     def has_cells_of_type(self, name, entity=False):
#         if entity:
#             with self.open() as resource:
#                 return name in list(resource()["/entities"])
#         else:
#             with self.open() as resource:
#                 return name in list(resource()["/cells"].attrs["types"])
#
#     def get_cells_of_type(self, name, entity=False):
#         # Check if cell type is present
#         if not self.has_cells_of_type(name, entity=entity):
#             raise DatasetNotFoundError(
#                 "Attempting to load {} type '{}' that isn't defined in the storage.".format(
#                     "cell" if not entity else "entity", name
#                 )
#             )
#         if entity:
#             with self.open() as resource:
#                 return resource()["/entities/" + name][()]
#         # Slice out the cells of this type based on the map in the position dataset attributes.
#         with self.open() as resource:
#             type_map = self.get_type_map(name)
#             return resource()["/cells/positions"][()][type_map]
#
#     def get_type_map(self, type):
#         with self.open() as resource:
#             return resource()["/cells/type_maps/{}_map".format(type)][()]
#
#     def get_connectivity_set_connection_types(self, tag):
#         """
#             Return all the ConnectionStrategies that contributed to the creation of this
#             connectivity set.
#         """
#         with self.open() as f:
#             # Get list of contributing types
#             type_list = f()["cells/connections/" + tag].attrs["connection_types"]
#             # Map contributing type names to contributing types
#             return list(
#                 map(lambda name: self.scaffold.get_connection_type(name), type_list)
#             )
#
#     def get_connectivity_set_meta(self, tag):
#         """
#             Return the metadata associated with this connectivity set.
#         """
#         with self.open() as f:
#             return dict(f()["cells/connections/" + tag].attrs)
#
#     def get_connectivity_set(self, tag):
#         return ConnectivitySet(self, tag)
#
#     def get_placement_set(self, tag):
#         return PlacementSet(self, tag)
#
#     @classmethod
#     def reconfigure(cls, hdf5_file, config):
#         if not os.path.exists(hdf5_file):
#             raise FileNotFoundError(
#                 "HDF5 file '{}' to reconfigure does not exist.".format(hdf5_file)
#             )
#         hdf5_formatter = cls()
#         hdf5_formatter.file = hdf5_file
#         hdf5_formatter.store_configuration(config)
