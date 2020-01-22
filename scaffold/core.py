from .statistics import Statistics
from .plotting import plot_network
import numpy as np
import time
from .trees import TreeCollection
from .output import MorphologyRepository
from .helpers import map_ndarray, listify_input
from .models import CellType
from .connectivity import ConnectionStrategy
from warnings import warn as std_warn

###############################
## Scaffold class
#    * Bootstraps configuration
#    * Loads geometries, morphologies, ...
#    * Creates network architecture
#    * Sets up simulation


class TreeCollectionGroup:
    def add_collection(self, name, handler):
        self.__dict__[name] = TreeCollection(name, handler)


def from_hdf5(file, verbosity=1):
    """
        Generate a :class:`Scaffold` from an HDF5 file.

        :param file: Path to the HDF5 file.
        :returns: A scaffold object
        :rtype: :class:`Scaffold`
    """
    from .config import _from_hdf5

    config = _from_hdf5(file, verbosity)
    return Scaffold(config, from_file=file)


class Scaffold:
    """
        This is the main object of the dbbs-scaffold package and bootstraps itself
        with a :doc:`configuration </configuration>`.

        During the compilation phase it can :doc:`place </placement>` and
        :doc:`connect </connectivity>` cells based on Layers,
        :doc:`cell type </configuration/cell-type>` and :doc:`connection type
        </configuration/connection-type>` configuration.

        The output can be stored in different :doc:`formats </output/formats>` and
        can be used to have the scaffold set up simulations in common neuroscience
        simulators such as NEST or NEURON.
    """

    def __init__(self, config, from_file=None):
        self.configuration = config
        self.reset_network_cache()
        # Debug statistics, unused.
        self.statistics = Statistics(self)
        self._initialise_output_formatter()
        self.trees = TreeCollectionGroup()
        self.trees.add_collection("cells", self.output_formatter)
        self.trees.add_collection("morphologies", self.output_formatter)
        self._nextId = 0
        # Use the configuration to initialise all components such as cells and layers
        # to prepare for the network architecture compilation.
        self.initialiseComponents()
        self.initialiseSimulators()

        # Tell the output formatter that we've loaded from an output and initialise scaffold from it.
        if from_file:
            self.output_formatter.file = from_file
            self.output_formatter.init_scaffold()

    def initialiseComponents(self):
        # Initialise the components now that the scaffoldInstance is available
        self._initialise_layers()
        self._initialise_cells()
        self._initialise_morphologies()
        self._initialise_placement_strategies()
        self._initialise_connection_types()
        self._initialise_simulations()

    def report(self, message, level=2, ongoing=False):
        if self.configuration.verbosity >= level:
            print(message, end="\n" if not ongoing else "\r")

    def warn(self, message, category=None):
        if self.configuration.verbosity > 0:
            std_warn(message, category, stacklevel=2)

    def initialiseSimulators(self):
        self.simulators = self.configuration.simulators

    def _initialise_cells(self):
        for cell_type in self.configuration.cell_types.values():
            cell_type.initialise(self)

    def _initialise_layers(self):
        for layer in self.configuration.layers.values():
            layer.initialise(self)

    def _initialise_placement_strategies(self):
        for placement in self.configuration.placement_strategies.values():
            placement.initialise(self)

    def _initialise_connection_types(self):
        for connection_type in self.configuration.connection_types.values():
            connection_type.initialise(self)

    def _initialise_morphologies(self):
        for geometry in self.configuration.morphologies.values():
            geometry.initialise(self)

    def _initialise_simulations(self):
        for simulation in self.configuration.simulations.values():
            self._initialise_simulation(simulation)

    def _initialise_simulation(self, simulation):
        simulation.initialise(self)
        for sim_cell in simulation.cell_models.values():
            sim_cell.initialise(self)
        for sim_connection in simulation.connection_models.values():
            sim_connection.initialise(self)
        for stimulus in simulation.devices.values():
            stimulus.initialise(self)

    def place_cell_types(self):
        sorted_cell_types = CellType.resolve_order(self.configuration.cell_types)
        for cell_type in sorted_cell_types:
            # Place cell type according to PlacementStrategy
            cell_type.placement.place()
            if cell_type.entity:
                entities = self.entities_by_type[cell_type.name]
                self.report(
                    "Finished placing {} {} entities.".format(
                        len(entities), cell_type.name
                    ),
                    2,
                )
            else:
                # Get the placed cells
                cells = self.cells_by_type[cell_type.name][:, 2:5]
                # Construct a tree of the placed cells
                self.trees.cells.create_tree(cell_type.name, cells)
                self.report(
                    "Finished placing {} {} cells.".format(len(cells), cell_type.name), 2
                )

    def connect_cell_types(self):
        sorted_connection_types = ConnectionStrategy.resolve_order(
            self.configuration.connection_types
        )
        for connection_type in sorted_connection_types:
            connection_type.connect()
            # Iterates for each tag of the connection_type
            for tag in range(len(connection_type.tags)):
                conn_num = np.shape(connection_type.get_connection_matrices()[tag])[0]
                source_name = connection_type.from_cell_types[0].name
                target_name = connection_type.to_cell_types[0].name
                self.report(
                    "Finished connecting {} with {} (tag: {} - total connections: {}).".format(
                        source_name, target_name, connection_type.tags[tag], conn_num
                    ),
                    2,
                )

    def run_after_placement_hooks(self):
        for hook in self.after_placement_hooks:
            hook.after_placement()

    def compile_network(self, tries=1, output=True):
        times = np.zeros(tries)
        # Place the cells starting from the lowest density cell_types.
        for i in np.arange(tries, dtype=int):
            if i > 0:
                self.reset_network_cache()
            t = time.time()
            self.place_cell_types()
            self.after_placement_hooks()
            self.connect_cell_types()
            times[i] = time.time() - t

            if output:
                self.compile_output()

            for type in self.configuration.cell_types.values():
                if type.entity:
                    count = self.entities_by_type[type.name].shape[0]
                else:
                    count = self.cells_by_type[type.name].shape[0]
                placed = type.placement.get_placement_count()
                if placed == 0 or count == 0:
                    self.report("0 {} placed (0%)".format(type.name), 1)
                    continue
                density_msg = ""
                percent = int((count / type.placement.get_placement_count()) * 100)
                if type.placement.layer is not None:
                    volume = type.placement.layer_instance.volume
                    density_gotten = "%.4g" % (count / volume)
                    density_wanted = "%.4g" % (
                        type.placement.get_placement_count() / volume
                    )
                    density_msg = " Desired density: {}. Actual density: {}".format(
                        density_wanted, density_gotten
                    )
                self.report(
                    "{} {} placed ({}%).".format(count, type.name, percent,), 2,
                )
            self.report("Average runtime: {}".format(np.average(times)), 2)

    def _initialise_output_formatter(self):
        self.output_formatter = self.configuration.output_formatter
        self.output_formatter.initialise(self)
        # Alias the output formatter to some other functions it provides.
        self.morphology_repository = self.output_formatter
        self.tree_handler = self.output_formatter
        # Load an actual morphology repository if it is provided
        if (
            not self.is_compiled()
            and self.output_formatter.morphology_repository is not None
        ):
            # We are in a precompilation state and the configuration specifies us to use a morpho repo.
            self.morphology_repository = MorphologyRepository(
                self.output_formatter.morphology_repository
            )

    def plot_network_cache(self):
        plot_network(self, from_memory=True)

    def reset_network_cache(self):
        # Cell positions dictionary per cell type. Columns: X, Y, Z.
        cell_types = list(
            filter(
                lambda c: not hasattr(c, "entity") or not c.entity,
                self.configuration.cell_types.values(),
            )
        )
        entities = list(
            filter(
                lambda c: hasattr(c, "entity") and c.entity,
                self.configuration.cell_types.values(),
            )
        )
        self.cells_by_type = {c.name: np.empty((0, 5)) for c in cell_types}
        # Entity IDs per cell type.
        self.entities_by_type = {e.name: np.empty((0)) for e in entities}
        # Cell positions dictionary per layer. Columns: Type, X, Y, Z.
        self.cells_by_layer = {
            key: np.empty((0, 5)) for key in self.configuration.layers.keys()
        }
        # Cells collection. Columns: Cell ID, Type, X, Y, Z.
        self.cells = np.empty((0, 5))
        # Cell connections per connection type. Columns: From ID, To ID.
        self.cell_connections_by_tag = {
            key: np.empty((0, 2)) for key in self.configuration.connection_types.keys()
        }
        self.connection_morphologies = {}
        self.connection_compartments = {}
        self.appends = {}
        self.placement_stitching = []
        self._connectivity_set_meta = {}
        self.labels = {}

    def run_simulation(self, simulation_name):
        simulation, simulator = self.prepare_simulation(simulation_name)
        simulation.simulate(simulator)
        return simulation

    def get_simulation(self, simulation_name):
        if simulation_name not in self.configuration.simulations:
            raise Exception(
                "Unknown simulation '{}', choose from: {}".format(
                    simulation_name, ", ".join(self.configuration.simulations.keys())
                )
            )
        simulation = self.configuration.simulations[simulation_name]
        return simulation

    def prepare_simulation(self, simulation_name, hdf5=None):
        simulation = self.get_simulation(simulation_name)
        simulator = simulation.prepare()
        return simulation, simulator

    def place_cells(self, cell_type, layer, positions):
        cell_count = positions.shape[0]
        if cell_count == 0:
            return
        # Create an ID for each cell.
        cell_ids = self._allocate_ids(positions.shape[0])
        # Store cells as ID, typeID, X, Y, Z
        cell_data = np.column_stack(
            (cell_ids, np.ones(positions.shape[0]) * cell_type.id, positions)
        )
        # Cache them per type
        self.cells_by_type[cell_type.name] = np.concatenate(
            (self.cells_by_type[cell_type.name], cell_data)
        )
        # Cache them per layer
        self.cells_by_layer[layer.name] = np.concatenate(
            (self.cells_by_layer[layer.name], cell_data)
        )
        # Store
        self.cells = np.concatenate((self.cells, cell_data))

        placement_dict = self.statistics.cells_placed
        if cell_type.name not in placement_dict:
            placement_dict[cell_type.name] = 0
        placement_dict[cell_type.name] += cell_count
        if not hasattr(cell_type.placement, "cells_placed"):
            cell_type.placement.__dict__["cells_placed"] = 0
        cell_type.placement.cells_placed += cell_count
        # Keep track of the order of placement, so that it can be emulated in simulators
        self.placement_stitching.append((cell_type.id, cell_ids[0], cell_count))

    def _allocate_ids(self, count):
        # Allocate a set of unique cell IDs in the scaffold.
        IDs = np.array(range(self._nextId, self._nextId + count), dtype=int)
        self._nextId += count
        return IDs

    def connect_cells(
        self,
        connection_type,
        connectome_data,
        tag=None,
        morphologies=None,
        compartments=None,
        meta=None,
    ):
        """
            Store connections for a connection type. Will store the
            ``connectome_data`` under ``scaffold.cell_connections_by_tag``, a
            mapped version of the morphology names under
            ``scaffold.connection_morphologies`` and the compartments under
            ``scaffold.connection_compartments``.

            :param connection_type: The connection type. The name of the connection type will be used by default as the tag.
            :type connection_type: :class:`ConnectionStrategy`
            :param connectome_data: A 2D ndarray with 2 columns: the presynaptic cell id and the postsynaptic cell id.
            :type connectome_data: :class:`numpy.ndarray`
            :param tag: The name of the dataset in the storage. If no tag is given, the name of the connection type is used. This parameter can be used to create multiple different connection set per connection type.
            :type tag: string
            :param morphologies: A 2D ndarray with 2 columns: the presynaptic morphology name and the postsynaptic morphology name.
            :type morphologies: :class:`numpy.ndarray`
            :param compartments: A 2D ndarray with 2 columns: the presynaptic compartment id and the postsynaptic compartment id.
            :type compartments: :class:`numpy.ndarray`
            :param meta: Additional metadata to be stored on the connectivity set.
            :type meta: dict
        """
        # Allow 1 connection type to store multiple connectivity datasets by utilizing tags
        tag = tag or connection_type.name
        # Keep track of relevant tags in the connection_type object
        if tag not in connection_type.tags:
            connection_type.tags.append(tag)
        self._append_tagged("cell_connections_by_tag", tag, connectome_data)
        if compartments is not None or morphologies is not None:
            if len(morphologies) != len(connectome_data) or len(compartments) != len(
                connectome_data
            ):
                raise Exception(
                    "The morphological data did not match the connectome data."
                )
            self._append_mapped("connection_morphologies", tag, morphologies)
            self._append_tagged("connection_compartments", tag, compartments)
        # Store the metadata internally until the output is compiled.
        if meta is not None:
            self._connectivity_set_meta[tag] = meta

    def create_entities(self, cell_type, n_cells_to_place):
        if n_cells_to_place == 0:
            return
        # Create an ID for each entity.
        entities_ids = self._allocate_ids(n_cells_to_place)

        # Cache them per type
        if not cell_type.name in self.entities_by_type:
            self.entities_by_type[cell_type.name] = entities_ids
        else:
            self.entities_by_type[cell_type.name] = np.concatenate(
                (self.entities_by_type[cell_type.name], entities_ids)
            )

        placement_dict = self.statistics.cells_placed
        if not cell_type.name in placement_dict:
            placement_dict[cell_type.name] = 0
        placement_dict[cell_type.name] += n_cells_to_place
        if not hasattr(cell_type.placement, "cells_placed"):
            cell_type.placement.__dict__["cells_placed"] = 0
        cell_type.placement.cells_placed += n_cells_to_place

    def _append_tagged(self, attr, tag, data):
        # Appends or creates data to a tagged numpy array in a dictionary attribute of the scaffold.
        if tag in self.__dict__[attr]:
            cache = self.__dict__[attr][tag]
            self.__dict__[attr][tag] = np.concatenate((cache, data))
        else:
            self.__dict__[attr][tag] = np.copy(data)

    def _append_mapped(self, attr, tag, data):
        # Appends or creates the data with a map to a tagged numpy array in a dictionary attribute of the scaffold.
        if not attr + "_map" in self.__dict__[attr]:
            self.__dict__[attr][tag + "_map"] = []
        mapped_data, data_map = map_ndarray(data, _map=self.__dict__[attr][tag + "_map"])
        mapped_data = np.array(mapped_data, dtype=int)
        if tag in self.__dict__[attr]:
            cache = self.__dict__[attr][tag]
            self.__dict__[attr][tag] = np.concatenate((cache, mapped_data))
        else:
            self.__dict__[attr][tag] = np.copy(mapped_data)

    def append_dset(self, name, data):
        self.appends[name] = data

    def get_cells_by_type(self, name):
        if name not in self.cells_by_type:
            raise Exception("Attempting to load unknown cell type '{}'".format(name))
        if self.cells_by_type[name].shape[0] == 0:
            if not self.output_formatter.exists():
                return self.cells_by_type[name]
            if self.output_formatter.has_cells_of_type(name):
                self.cells_by_type[name] = self.output_formatter.get_cells_of_type(name)
            else:
                raise Exception("Cell type '{}' not found in output storage".format(name))
        return self.cells_by_type[name]

    def get_entities_by_type(self, name):
        if name not in self.entities_by_type:
            raise Exception("Attempting to load unknown entity type '{}'".format(name))
        if self.entities_by_type[name].shape[0] == 0:
            if not self.output_formatter.exists():
                return self.entities_by_type[name]
            if self.output_formatter.has_cells_of_type(name, entity=True):
                self.entities_by_type[name] = self.output_formatter.get_cells_of_type(
                    name, entity=True
                )
            else:
                raise Exception(
                    "Entity type '{}' not found in output storage".format(name)
                )
        return self.entities_by_type[name]

    def compile_output(self):
        self.output_formatter.create_output()

    def _connection_types_query(self, postsynaptic=[], presynaptic=[]):
        # This function searches through all connection types that include the given
        # pre- and/or postsynaptic cell types.

        # Make sure the inputs are lists.
        postsynaptic = listify_input(postsynaptic)
        presynaptic = listify_input(presynaptic)

        # Local function that checks for any intersection between 2 lists based on a given f.
        def any_intersect(l1, l2, f=lambda x: x):
            if not l2:  # Return True if there's no pre/post targets specified
                return True
            for e1 in l1:
                if f(e1) in l2:
                    return True
            return False

        # Extract the connection types as tuples so that they can be turned back into a
        # dictionary after filtering
        connection_items = self.configuration.connection_types.items()
        # Lambda that includes any connection type with at least one of the specified
        # presynaptic and one of the specified postsynaptic connections.
        # If the post- or presynaptic constraints are empty all connection types pass for
        # that constraint.
        intersect = lambda c: any_intersect(
            c[1].to_cell_types, postsynaptic, lambda x: x.name
        ) and any_intersect(c[1].from_cell_types, presynaptic, lambda x: x.name)
        # Filter all connection types based on the lambda function.
        filtered_connection_items = list(filter(intersect, connection_items,))
        # Turn the filtered result into a dictionary.
        return dict(filtered_connection_items)

    def get_connection_types_by_cell_type(
        self, any=None, postsynaptic=None, presynaptic=None
    ):
        """
            Search for connection types that include specific cell types as pre- or postsynaptic targets.

            :param any: Cell type names that will include connection types that have the given cell types as either pre- or postsynaptic targets.
            :type any: string or sequence of strings.
            :param postsynaptic: Cell type names that will include connection types that have the given cell types as postsynaptic targets.
            :type postsynaptic: string or sequence of strings.
            :param presynaptic: Cell type names that will include connection types that have the given cell types as presynaptic targets.
            :type presynaptic: string or sequence of strings.
            :returns: The connection types that meet the specified criteria.
            :rtype: dict
        """
        if any is None and postsynaptic is None and presynaptic is None:
            raise ArgumentError("No cell types specified")
        # Make a list out of the input elements
        postsynaptic = listify_input(postsynaptic)
        presynaptic = listify_input(presynaptic)
        # Initialize empty omitted lists
        if any is not None:  # Add any cell types as both post and presynaptic targets
            any = listify_input(any)
            postsynaptic.extend(any)
            presynaptic.extend(any)
        # Execute the query and return results.
        return self._connection_types_query(postsynaptic, presynaptic)

    def get_connection_cache_by_cell_type(
        self, any=None, postsynaptic=None, presynaptic=None
    ):
        """
            Get the connections currently in the cache for connection types that include certain cell types as targets.

            :see: get_connection_types_by_cell_type
        """
        # Find the connection types that have the specified targets
        connection_types = self.get_connection_types_by_cell_type(
            any, postsynaptic, presynaptic
        )
        # Map them to a list of tuples with the 1st element the connection type
        # and the connection matrices appended behind it.
        return list(
            map(lambda x: (x, *x.get_connection_matrices()), connection_types.values())
        )

    def get_connections_by_cell_type(self, any=None, postsynaptic=None, presynaptic=None):
        """
            Get the connectivity sets from storage for connection types that include certain cell types as targets.

            :see: get_connection_types_by_cell_type
            :rtype: :class:`scaffold.models.ConnectivitySet`
        """
        # Find the connection types that have the specified targets
        connection_types = self.get_connection_types_by_cell_type(
            any, postsynaptic, presynaptic
        )
        # Map them to a list of tuples with the 1st element the connection type
        # and the connection matrices appended behind it.
        return list(
            map(lambda x: (x, *x.get_connectivity_sets()), connection_types.values())
        )

    def get_connectivity_set(self, tag):
        return self.output_formatter.get_connectivity_set(tag)

    def translate_cell_ids(self, data, cell_type):
        if not self.is_compiled():
            return self.cells_by_type[cell_type.name][data, 0]
        else:
            return np.array(self.output_formatter.get_type_map(cell_type))[data]

    def compile_output(self):
        self.output_formatter.create_output()

    def get_connection_type(self, name):
        if name not in self.configuration.connection_types:
            raise Exception("Unknown connection type '{}'".format(name))
        return self.configuration.connection_types[name]

    def get_cell_types(self, entities=True):
        """
            Return a collection of all configured cell types.

            ::

              for cell_type in scaffold.get_cell_types():
                  print(cell_type.name)
        """
        if entities:
            return self.configuration.cell_types.values()
        else:
            return list(filter(lambda c: not c.entity, self.get_cell_types()))

    def get_entity_types(self):
        """
            Return a list of connection types that describe entities instead
            of cells.
        """
        return list(
            filter(
                lambda t: hasattr(t, "entity") and t.entity is True,
                self.configuration.connection_types.values(),
            )
        )

    def get_cell_type(self, name):
        if name not in self.configuration.cell_types:
            raise Exception("Unknown cell type '{}'".format(name))
        return self.configuration.cell_types[name]

    def get_cell_position(self, id):
        if not id < len(self.cells):
            raise Exception(
                "Cell {} does not exist. (highest id is {})".format(
                    id, len(self.cells) - 1
                )
            )
        return self.cells[id, 2:5]

    def get_cell_positions(self, selector):
        return self.cells[selector, 2:5]

    def get_cells(self, selector):
        return self.cells[selector]

    def get_placed_count(self, cell_type_name):
        return self.statistics.cells_placed[cell_type_name]

    def is_compiled(self):
        return self.output_formatter.exists()

    def create_adapter(self, simulation_name):
        if simulation_name not in self.configuration.simulations:
            raise Exception("Unknown simulation '{}'".format(simulation_name))
        simulations = self.configuration._parsed_config["simulations"]
        simulation_config = simulations[simulation_name]
        adapter = self.configuration.init_simulation(
            simulation_name, simulation_config, return_obj=True
        )
        self.configuration.finalize_simulation(
            simulation_name, simulation_config, adapter
        )
        self._initialise_simulation(adapter)
        return adapter

    def label_cells(self, label, ids):
        if self.scaffold.labels == {}:
            scaffold.labels[label] = ids
        else:
            scaffold.labels[label] = np.append(scaffold.labels[label], ids)
