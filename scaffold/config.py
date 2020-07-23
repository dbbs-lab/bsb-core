"""
    This module handles all configuration of the scaffold.
"""

import os, sys, site
from inspect import isclass
from .models import CellType, Layer
from .morphologies import Morphology as BaseMorphology
from .connectivity import ConnectionStrategy, FiberTransform
from .placement import PlacementStrategy
from .output import OutputFormatter, HDF5Formatter
from .simulation import SimulatorAdapter, SimulationComponent
from .helpers import (
    assert_float,
    assert_attr_array,
    assert_attr_float,
    assert_attr,
    if_attr,
    assert_strictly_one,
    assert_attr_in,
    ConfigurableClass,
    load_configurable_class,
    fill_configurable_class,
    get_config_path,
)
from .simulators.nest import NestAdapter
from .postprocessing import PostProcessingHook
from .simulators.neuron import NeuronAdapter
from .exceptions import *
import numpy as np


def _from_hdf5(file):
    """
        Restore a configuration object from an HDF5 file.

        :param file: Path of the HDF5 file.
        :param verbosity: Verbosity level of the reporting
        :type file: string
        :rtype: None
    """
    import h5py

    # Open read only
    with h5py.File(file, "r") as resource:
        # Get the serialized configuration details
        config_class = resource.attrs["configuration_class"]
        config_string = resource.attrs["configuration_string"]
    # Determine the class and module name and import them
    class_parts = config_class.split(".")
    class_name = class_parts[-1]
    module_name = ".".join(class_parts[:-1])
    if module_name == "":
        module_dict = globals()
    else:
        module_dict = __import__(
            module_name, globals(), locals(), [class_name], 0
        ).__dict__
    if class_name not in module_dict:
        raise DynamicClassError(
            "Can not load HDF5 file '{}'. Configuration class not found:".format(file)
            + config_class
        )
    # Instantiate the configuration class with a configuration stream
    return module_dict[class_name](stream=config_string)


class ScaffoldConfig(object):
    """
        Main configuration object. Is passed into the scaffold constructor
        to configurate it.
    """

    def __init__(self, file=None, stream=None, simulators={}):
        """
            :param file: The path to a configuration file
            :type file: string
            :param stream: A string containing configuration content you'd find in a configuration file.
            :type stream: string
            :param verbosity: Sets the level of detail on the console feedback that the scaffold reports. 0: Errors only. 1: 0 + warnings. 2: 1 + updates. 3: 2 + progress
            :type verbosity: int
            :param simulators: Additional simulators to register. A dictionary of :class:`SimulatorAdapter`s
            :type simulators: dict
        """
        # Initialise empty config object.

        # Dictionaries and lists
        self.cell_types = {}
        self.cell_type_map = []
        self.layers = {}
        self.layer_map = []
        self.connection_types = {}
        self.morphologies = {}
        self.placement_strategies = {}
        self.after_placement_hooks = {}
        self.after_connect_hooks = {}
        self.simulations = {}
        self._raw = ""
        self._name = ""
        if not hasattr(self, "_extension"):
            self._extension = ""
        self.simulators = simulators
        self.simulators["nest"] = NestAdapter
        self.simulators["neuron"] = NeuronAdapter
        self.output_formatter = HDF5Formatter()

        # Fallback simulation values
        self.X = 200  # Transverse simulation space size (µm)
        self.Z = 200  # Longitudinal simulation space size (µm)

        if (
            file is not None or stream is not None
        ):  # Load config from file or stream of data
            self.read_config(file, stream)
            # Execute the load handler set by the child configuration implementation
            self._parsed_config = self._load_handler(self._raw)
            # Remove the load handler so that the scaffold can be pickled. (#76)
            delattr(self, "_load_handler")

    def read_config(self, file=None, stream=None):
        if stream is not None:
            self.read_config_stream(stream)
        else:
            self.read_config_file(file or "mouse_cerebellum_cortex")

    def read_config_file(self, file):
        # Determine extension of file.
        head, tail = os.path.splitext(file)
        # Append extension and send warning if extension is not present.
        if tail != self._extension:
            file += self._extension
        # Try to open the config file from the specified directory
        try:
            with open(file, "r") as file:
                self._raw = file.read()
                self._name = file.name
                return
        except Exception as e:
            pass
        # Try to open the config file from the default directory
        try:
            with open(get_config_path(file), "r") as file:
                self._raw = file.read()
                self._name = file.name
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Could not find the configuration file '{}' in the specified folder or included in the package.".format(
                    file
                )
            ) from None

    def read_config_stream(self, stream):
        self._raw = stream
        self._name = "<stream>"

    def add_cell_type(self, cell_type):
        """
            Adds a :class:`CellType` to the config object. Cell types are used
            to populate cells into the layers of the simulation.

            :param cell_type: CellType object to add
            :type cell_type: :class:`CellType`
        """
        # Register a new cell type model.
        self.cell_types[cell_type.name] = cell_type
        self.cell_type_map.append(cell_type.name)

    def add_morphology(self, morphology):
        """
            Adds a :class:`Morphology` to the config object. Morphologies are
            used to determine which cells can be connected to eachother.

            :param morphology: :class:`Morphology` object to add
            :type morphology: Morphology
        """
        # Register a new Geometry.
        self.morphologies[morphology.name] = morphology

    def add_placement_strategy(self, placement):
        """
            Adds a :class:`PlacementStrategy` to the config object. Placement
            strategies are used to place cells in the simulation volume.

            :param placement: :class:`PlacementStrategy` object to add
            :type placement: :class:`PlacementStrategy`
        """
        # Register a new Geometry.
        self.placement_strategies[placement.name] = placement

    def add_after_placement_hook(self, hook):
        """
            Adds a :class:`PostProcessingHook` to the config object. After placement has
            completed the class's ``after_placement`` function will be called.

            :param hook: :class:`PostProcessingHook`
        """
        # Register a new Geometry.
        self.after_placement_hooks[hook.name] = hook

    def add_after_connectivity_hook(self, hook):
        """
            Adds a :class:`PostProcessingHook` to the config object. After connectivity has
            completed the class's ``after_connectivity`` function will be called.

            :param hook: :class:`PostProcessingHook`
        """
        # Register a new Geometry.
        self.after_connect_hooks[hook.name] = hook

    def add_connection(self, connection):
        """
            Adds a :class:`ConnectionStrategy` to the config object. ConnectionStrategies
            are used to determine which touching cells to connect.

            :param connection: ConnectionStrategy object to add
            :type connection: ConnectionStrategy
        """
        # Register a new ConnectionStrategy.
        self.connection_types[connection.name] = connection

    def add_simulation(self, simulation):
        """
            Adds a :class:`SimulatorAdapter` to the config object. S
            are used to determine which touching cells to connect.

            :param connection: :class:`ConnectionStrategy` object to add
            :type connection: :class:`ConnectionStrategy`
        """
        # Register a new simulation
        self.simulations[simulation.name] = simulation

    def add_layer(self, layer):
        """
            Adds a :class:`Layer` to the config object. layers are regions of
            the simulation to be populated by cells.

            :param layer: :class:`Layer` object to add
            :type layer: :class:`Layer`
        """
        # Register a new layer model.
        self.layers[layer.name] = layer
        self.layer_map.append(layer.name)

    def get_layer(self, name="", id=-1):
        """
            Finds a layer by its name or id.

            :param name: Name of the layer to look for.
            :type name: string
            :param id: Id of the layer to look for.
            :type id: int

            :returns: A :class:`Layer`: object.
            :rtype: :class:`Layer`
            :raises: **Exception:** If the name or id isn't found.
                **ArgumentError:** If no name or id is provided.
        """
        if id > -1:
            if len(self.layer_map) <= id:
                raise LayerNotFoundError("Layer with id {} not found.".format(id))
            return list(self.layers.values())[id]
        if name != "":
            if name not in self.layers:
                raise LayerNotFoundError("Layer with name '{}' not found".format(name))
            return self.layers[name]
        raise ArgumentError(
            "Invalid arguments for ScaffoldConfig.get_layer: name='{}', id={}".format(
                name, id
            )
        )

    def get_layer_id(self, name):
        """
            Get the ID of a layer.

            :param name: Name of the layer
            :type name: string
            :returns: The ID of the layer
            :rtype: int
            :raises: **ValueError** if the name is not found.
        """
        return self.layer_map.index(name)

    def get_layer_list(self):
        """
            Get a list of the layers.

            :returns: A list of :class:`Layer`.
            :rtype: list
        """
        return list(self.layers.values())

    def get_cell_type(self, name):
        """
            Get cell type.

            :param name: Name or id of the cell type.
            :type name: string
            :returns: The cell type with given identifier.
            :rtype: :class:`CellType`
        """
        if isinstance(name, str):
            if name in self.cell_types:
                return self.cell_types[name]
            else:
                raise TypeNotFoundError(
                    "Cell type with name '{}' not found.".format(name)
                )
        # `name` is not a string, so use it as an ID in the cell type map
        # to obtain the actual name.
        return self.cell_types[self.cell_type_map[name]]

    def resize(self, X=None, Z=None):
        """
            Resize the configured simulation volume.

            .. warning::
                Do not change the ``X`` or ``Z`` values directly as it will not
                resize all :class:`Layer` objects and cause undefined behavior.

            :param X: The new X size.
            :type X: float
            :param Z: The new Z size.
            :type Z: float
            :rtype: None
        """

        scaling_x = 1.0
        scaling_z = 1.0
        if X is not None:
            scaling_x = X / self.X
            self.X = X
        if Z is not None:
            scaling_z = Z / self.Z
            self.Z = Z
        for layer in self.layers.values():
            if layer.scaling:
                layer.dimensions[0] *= scaling_x
                layer.dimensions[2] *= scaling_z
            if layer.xz_center:
                layer.origin[0] = self.X / 2 - layer.dimensions[0] / 2
                layer.origin[2] = self.Z / 2 - layer.dimensions[2] / 2
        # Scale layers that depend on the size of other layers.
        self.scale_layers()

    def scale_layers(self):
        for layer in self.layers.values():
            if hasattr(layer, "volume_scale"):
                layer.scale_to_reference()


class JSONConfig(ScaffoldConfig):
    """
        Create a scaffold configuration from a JSON formatted file/string.
    """

    def __init__(self, file=None, stream=None, **kwargs):
        """
            Initialize config from .json file.

            :param file: Path of the configuration .json file.
            :type file: string
            :param stream: INI formatted string representing the configuration file.
            :type file: string
            :param verbosity: Verbosity (output level) of the scaffold.
            :type file: int
            :param simulators: Dictionary with extra simulators to register
            :type simulators: {string: :class:`SimulatorAdapter`}
        """

        def load_handler(config_string):
            # Use the JSON module to parse the configuration string.
            import json

            try:
                return json.loads(config_string)
            except json.decoder.JSONDecodeError as e:
                raise ConfigurationFormatError(
                    "Error while loading JSON configuration: {}".format(e)
                ) from None

        # Set flags to indicate we expect a json configuration.
        self._type = "json"
        self._extension = ".json"
        # Tells the base configuration class how to parse the configuration string
        self._load_handler = load_handler
        if file is None and stream is None:
            file = "mouse_cerebellum_cortex"
        # Initialize base config class, handling the reading of file/stream to string
        ScaffoldConfig.__init__(self, file=file, stream=stream, **kwargs)

        # Use the parsed configuration as a basis for loading all parts of the scaffold
        parsed_config = self._parsed_config
        # Load the general scaffold configuration
        self.load_general(parsed_config)
        # Load the output module configuration
        self.load_output(parsed_config)
        self._layer_stacks = {}
        # Load the layers
        self.load_attr(
            config=parsed_config,
            attr="layers",
            init=self.init_layer,
            final=self.finalize_layers,
            single=True,
        )
        # Load the cell types
        self.load_attr(
            config=parsed_config,
            attr="cell_types",
            init=self.init_cell_type,
            final=self.finalize_cell_type,
        )
        # Load the after placement hooks
        self.load_attr(
            config=parsed_config,
            attr="after_placement",
            init=self.init_after_placement_hook,
            optional=True,
        )
        # Load the after connectivity hooks
        self.load_attr(
            config=parsed_config,
            attr="after_connectivity",
            init=self.init_after_connectivity_hook,
            optional=True,
        )
        # Load the connection types
        self.load_attr(
            config=parsed_config,
            attr="connection_types",
            init=self.init_connection,
            final=self.finalize_connection,
        )
        # Load the simulations
        self.load_attr(
            config=parsed_config,
            attr="simulations",
            init=self.init_simulation,
            final=self.finalize_simulation,
        )

    def load_general(self, config):
        """
            Load the general segment in a JSON configuration file.
        """
        if "network_architecture" not in config:
            raise ConfigurationError(
                "Missing 'network_architecture' attribute in configuration."
            )
        netw_config = config["network_architecture"]
        if "simulation_volume_x" not in netw_config:
            raise ConfigurationError(
                "Missing 'simulation_volume_x' attribute in 'network_architecture' configuration."
            )
        if "simulation_volume_z" not in netw_config:
            raise ConfigurationError(
                "Missing 'simulation_volume_x' attribute in 'network_architecture' configuration."
            )
        self.X = float(netw_config["simulation_volume_x"])
        self.Z = float(netw_config["simulation_volume_z"])

    def load_output(self, config):
        """
            Load the output segment in a JSON configuration file.
        """
        if "output" not in config:
            raise ConfigurationError("Missing 'output' attribute in configuration.")
        output_config = config["output"]
        if "format" not in output_config:
            raise ConfigurationError(
                "Missing 'format' attribute in 'output' configuration."
            )
        self.output_formatter = load_configurable_class(
            "output_formatter", output_config["format"], OutputFormatter
        )
        fill_configurable_class(self.output_formatter, output_config, excluded=["format"])

    def load_attr(
        self, config, attr, init, final=None, single=False, node_name=None, optional=False
    ):
        """
            Initialize and finalize a collection of children that are contained
            within a dictionary which is in turn inside of the given `config`
            node under the `attr` key.

            :param config: Parent configuration section that contains the
                dictionary of child definitions.
            :type config: dict
            :param attr: Key of the dictionary that contains the child
                definitions.
            :type attr: string
            :param init: The initializer function that turns each child
                definition into a semi-configured child object. Takes 2
                arguments: name and child configuration section.
            :type init: function
            :param final: The finalizer function that completes configuration of
                each child definition after they've all been initialized. Takes
                0 (if `single`=`True`) or 2 arguments: name and child
                configuration section.
            :type final: function
            :param single: Should the finalizer be called for each child or a
                single time? This allows you more control over the finalization
                process.
            :type single: bool
            :param node_name: The full location of the root node inside of the
                configuration file. This gives more accurate exceptions.
            :type node_name: string

        """
        if attr not in config:
            if optional:
                return
            raise ConfigurationError(
                "Missing '{}' attribute in {}.".format(attr, node_name or "configuration")
            )
        for def_name, def_config in config[attr].items():
            init(def_name, def_config)
        if single and final:
            final()
        elif final:
            for def_name, def_config in config[attr].items():
                final(def_name, def_config)

    def init_cell_type(self, name, section):
        """
            Initialise a CellType from a configuration object.

            :param section: A section of a configuration file, parsed by configparser.
            :type section: /

            :returns: A :class:`CellType` object.
            :rtype: :class:`CellType`
        """
        cell_type = CellType(name)
        node_name = "cell_types.{}".format(name)

        # Placement configuration
        placement_kwargs = {}
        # Get the placement configuration node
        placement = assert_attr(section, "placement", node_name)
        cell_type.placement = self.init_placement(placement, cell_type)
        cell_type.entity = "entity" in section and bool(section["entity"])
        cell_type.relay = "relay" in section and bool(section["relay"])
        if not cell_type.entity:
            # Get the morphology configuration node
            morphology = assert_attr(section, "morphology", node_name)
            cell_type.morphology = self.init_morphology(morphology, name)

        cell_type.plotting = PlottingConfig(name, color="#000000")
        if "plotting" in section:
            cell_type.plotting.label = if_attr(section["plotting"], "display_name", name)
            cell_type.plotting.color = if_attr(section["plotting"], "color", "#000000")
            cell_type.plotting.opacity = if_attr(section["plotting"], "opacity", 1.0)
        # Register cell type
        self.add_cell_type(cell_type)
        return cell_type

    def init_layer(self, name, config):
        """
            Initialise a Layer from a json object.

            :param config: An object in the 'layers' array of the JSON file.
            :type config: /

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        """
        # Get thickness of the layer
        if "thickness" not in config and "volume_scale" not in config:
            raise ConfigurationError(
                "Either a thickness attribute or volume_scale required in {} config.".format(
                    name
                )
            )
        thickness = 0.0
        if "thickness" in config:
            thickness = float(config["thickness"])

        # Set the position of this layer in the space.
        if "position" not in config:
            origin = [0.0, 0.0, 0.0]
        else:
            # TODO: Catch possible casting errors to float.
            origin = [float(coord) for coord in config["position"]]
            if len(origin) != 3:
                raise ConfigurationError(
                    "Invalid position '{}' given in config '{}'".format(
                        config["position"], name
                    )
                )

        # Parse the layer stack config
        if "stack" in config:
            stack_config = config["stack"]
            if "stack_id" not in stack_config:
                raise ConfigurationError(
                    "A 'stack_id' attribute is required in '{}.stack'.".format(name)
                )
            stack_id = int(stack_config["stack_id"])
            stack = {"layers": {}}
            # Get or add stack to layer_stacks
            if stack_id in self._layer_stacks:
                stack = self._layer_stacks[stack_id]
            else:
                self._layer_stacks[stack_id] = stack
            if "position_in_stack" not in stack_config:
                raise ConfigurationError(
                    "A 'position_in_stack' attribute is required in '{}.stack'.".format(
                        name
                    )
                )
            stack_index = int(stack_config["position_in_stack"])
            if stack_index in stack["layers"]:
                raise ConfigurationError(
                    "Stack position {} already occupied by {} in stack {}".format(
                        stack_index, stack["layers"][stack_index], stack_id
                    )
                )
            else:
                stack["layers"][stack_config["position_in_stack"]] = name
            # Configurate the position of the stack
            if "position" in stack_config:
                if "position" in stack:
                    raise ConfigurationError(
                        "Duplicate positioning attribute found for stack with id '{}'".format(
                            stack_id
                        )
                    )
                stack["position"] = stack_config["position"]
        # Set the layer dimensions
        xzScale = [1.0, 1.0]
        # Scale by the XZ-scaling factor, if present
        if "xz_scale" in config:
            if not isinstance(config["xz_scale"], list):  # Not a list?
                # Try to convert it to a float and make a 2 element list out of it
                try:
                    xzScaleValue = float(config["xz_scale"])
                except Exception as e:
                    raise ConfigurationError(
                        "Could not convert xz_scale value '{}' to a float.".format(
                            config["xz_scale"]
                        )
                    )
                config["xz_scale"] = [xzScaleValue, xzScaleValue]
            xzScale[0] = float(config["xz_scale"][0])
            xzScale[1] = float(config["xz_scale"][1])
        dimensions = [self.X * xzScale[0], thickness, self.Z * xzScale[1]]
        #   and center the layer on the XZ plane, if present
        if "xz_center" in config and config["xz_center"] is True:
            origin[0] = (self.X - dimensions[0]) / 2.0
            origin[2] = (self.Z - dimensions[2]) / 2.0
        # Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        layer.xz_center = "xz_center" in config and config["xz_center"] is True
        # Add layer to the config object
        self.add_layer(layer)
        return layer

    def init_morphology(self, section, cell_type_name):
        """
            Initialize a Geometry-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        """
        name = cell_type_name + "_morphology"
        node_name = "cell_types.{}.morphology".format(cell_type_name)
        morphology_class = assert_attr(section, "class", node_name)
        morphology = load_configurable_class(name, morphology_class, BaseMorphology)
        fill_configurable_class(morphology, section, excluded=["class"])
        self.add_morphology(morphology)
        return morphology

    def init_connection(self, name, section):
        """
            Initialize a ConnectionStrategy-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        """
        node_name = "connection_types.{}".format(name)
        connection_class = assert_attr(section, "class", node_name)
        connection = load_configurable_class(name, connection_class, ConnectionStrategy)
        fill_configurable_class(
            connection,
            section,
            excluded=[
                "class",
                "from_cell_types",
                "to_cell_types",
                "simulation",
                "transform",
            ],
        )
        connection.__dict__["_from_cell_types"] = assert_attr_array(
            section, "from_cell_types", node_name
        )
        connection.__dict__["_to_cell_types"] = assert_attr_array(
            section, "to_cell_types", node_name
        )

        if "transform" in section:
            transform_class = assert_attr(
                section["transform"], "class", node_name + ".transform"
            )
            transformation = load_configurable_class(
                name + "_transform", transform_class, FiberTransform
            )
            fill_configurable_class(
                transformation, section["transform"], excluded=["class"],
            )
            connection.transformation = transformation
        else:
            connection.transformation = None

        self.add_connection(connection)

    def init_placement(self, section, cell_type):
        """
            Initialize a PlacementStrategy-subclass from the configuration. Uses __import__
            to fetch placement class, then copies all keys as is from config section to instance
            and adds it to the PlacementStrategies dictionary.
        """
        name = cell_type.name + "_placement"
        node_name = "cell_types.{}.placement".format(cell_type.name)
        placement_class = assert_attr(section, "class", node_name)
        try:
            placement = load_configurable_class(
                name,
                placement_class,
                PlacementStrategy,
                parameters={"cell_type": cell_type},
            )
        except ConfigurableClassNotFoundError as e:
            raise ConfigurableClassNotFoundError(
                "Couldn't find class '{}' specified in '{}'".format(
                    placement_class, node_name
                )
            )
        if not placement.is_entities():
            # Radius of the cell soma
            placement.soma_radius = assert_attr_float(section, "soma_radius", node_name)
            placement.radius = placement.soma_radius  # Alias it to the radius shorthand.
        # Density configurations all rely on a float or a float and relation
        density_attr, density_value = assert_strictly_one(
            section,
            [
                "density",
                "planar_density",
                "placement_count_ratio",
                "density_ratio",
                "count",
                "per_planet",
            ],
            node_name,
        )
        density_value = assert_float(
            density_value, "{}.{}".format(node_name, density_attr)
        )
        placement.__dict__[density_attr] = density_value
        # Does this density configuration rely on a relation to another cell_type?
        ratio_attrs = ["placement_count_ratio", "density_ratio"]
        if density_attr in ratio_attrs:
            relation = assert_attr(section, "placement_relative_to", node_name)
            placement.placement_relative_to = relation

        # Copy other information to be validated by the placement class
        fill_configurable_class(
            placement,
            section,
            excluded=[
                "class",
                "soma_radius",
                "density",
                "planar_density",
                "placement_count_ratio",
                "density_ratio",
                "per_planet",
            ],
        )

        # Register the configured placement class
        self.add_placement_strategy(placement)
        return placement

    def init_after_placement_hook(self, name, section):
        """
            Initialize an after-placement hook from the configuration.
        """
        node_name = "after_placement." + name
        hook_class = assert_attr(section, "class", node_name)
        hook = load_configurable_class(name, hook_class, PostProcessingHook)
        fill_configurable_class(hook, section, excluded=["class"])
        self.add_after_placement_hook(hook)
        return hook

    def init_after_connectivity_hook(self, name, section):
        """
            Initialize an after-connectivity hook from the configuration.
        """
        node_name = "after_connectivity." + name
        hook_class = assert_attr(section, "class", node_name)
        hook = load_configurable_class(name, hook_class, PostProcessingHook)
        fill_configurable_class(hook, section, excluded=["class"])
        self.add_after_connectivity_hook(hook)
        return hook

    def init_simulation(self, name, section, return_obj=False):
        """
            Initialize a simulation from a configuration node. Also initializes
            the subcomponents: `cell_models`, `connection_models` and `devices`.
        """
        node_name = "simulations.{}".format(name)
        # Get the simulator name from the config
        simulator_name = assert_attr_in(
            section, "simulator", self.simulators.keys(), node_name
        )
        # Get the simulator adapter class for this simulation
        simulator = self.simulators[simulator_name]
        # Initialise a new simulator adapter for this simulation
        simulation = load_configurable_class(name, simulator, SimulatorAdapter)
        # Configure the simulation's adapter
        fill_configurable_class(
            simulation,
            section,
            excluded=[
                "simulator",
                "cell_models",
                "connection_models",
                "devices",
                "entities",
            ],
        )
        # Get the classes required to configure cells and connections in this simulation
        config_classes = simulation.get_configuration_classes()

        # Factory that produces initialization functions for the simulation components
        def init_component_factory(component_type):
            component_class = config_classes[component_type]

            def init_component(component_name, component_config):
                component = self.init_simulation_component(
                    component_name, component_config, component_class, simulation
                )
                component.simulation = simulation
                component.node_name = (
                    "simulations." + simulation.name + "." + component_type
                )
                simulation.__dict__[component_type][component_name] = component

            # Return the initialization function
            return init_component

        # Load the simulations' cell models, connection models and devices from the configuration.
        self.load_attr(
            config=section,
            attr="cell_models",
            init=init_component_factory("cell_models"),
            node_name=node_name,
        )
        self.load_attr(
            config=section,
            attr="connection_models",
            init=init_component_factory("connection_models"),
            node_name=node_name,
        )
        self.load_attr(
            config=section,
            attr="devices",
            init=init_component_factory("devices"),
            node_name=node_name,
        )
        if "entities" in section:
            self.load_attr(
                config=section,
                attr="entities",
                init=init_component_factory("entities"),
                node_name=node_name,
            )

        if return_obj:
            return simulation
        # Add the simulation into the configuration
        self.add_simulation(simulation)

    def finalize_simulation(self, simulation_name, section, simulation=None):
        if simulation is None:
            simulation = self.simulations[simulation_name]
        pass

    def finalize_layers(self):
        # Parse layer scaling config
        for layer in self.layers.values():
            config = self._parsed_config["layers"][layer.name]
            if "volume_scale" in config:
                layer.volume_scale = config["volume_scale"]
                # Check if the config file specifies with respect to which layer volumes
                # we are scaling the current volume
                if "scale_from_layers" not in config:
                    raise ConfigurationError(
                        "Required attribute scale_from_layers missing in {} config.".format(
                            name
                        )
                    )
                layer.reference_layers = [
                    self.layers[l] for l in config["scale_from_layers"]
                ]
                # Ratio between dimensions (x,y,z) of the layer volume; if not specified, the layer is a cube
                dimension_ratios = [1.0, 1.0, 1.0]
                if "volume_dimension_ratio" in config:
                    dimension_ratios = config["volume_dimension_ratio"]
                # Normalize dimension ratios to y dimension
                layer.dimension_ratios = np.array(dimension_ratios) / dimension_ratios[1]
                layer.scale_to_reference()
        # Update layer stacks: set layer Y positions.
        self.layout_stacks()

    def layout_stacks(self):
        """
            Arrange layers according to their stack specifications.
        """
        for stack in self._layer_stacks.values():
            if "position" not in stack:
                stack["position"] = [0.0, 0.0, 0.0]
            # Get the current roof of the stack
            stack_roof = stack["position"][1]
            for (id, name) in sorted(stack["layers"].items()):
                layer = self.layers[name]
                # Place the layer on top of the roof of the stack, and move up the roof by
                # the thickness of the stacked layer.
                layer.origin[1] = stack_roof
                if layer.xz_center:
                    layer.origin[0] = stack["position"][0] + (
                        (self.X - layer.dimensions[0]) / 2.0
                    )
                    layer.origin[2] = stack["position"][2] + (
                        (self.Z - layer.dimensions[2]) / 2.0
                    )
                else:
                    layer.origin[0] = stack["position"][0]
                    layer.origin[2] = stack["position"][2]
                stack_roof += layer.thickness

    def finalize_cell_type(self, cell_type_name, section):
        """
            Finalize configuration of the cell type.
        """
        pass

    def finalize_connection(self, connection_name, section):
        node_name = "connection_types.{}"
        connection = self.connection_types[connection_name]
        from_cell_types = []
        from_cell_compartments = []
        to_cell_types = []
        to_cell_compartments = []
        i = 0
        for connected_cell in connection._from_cell_types:
            type = assert_attr(connected_cell, "type", node_name + ".{}".format(i))
            i += 1
            if type not in self.cell_types:
                raise ConfigurationError(
                    "Unknown cell type '{}' in '{}.from_cell_types'".format(
                        type, node_name
                    )
                )
            from_cell_types.append(self.cell_types[type])
            if "compartments" in connected_cell:
                from_cell_compartments.append(connected_cell["compartments"])
            else:
                from_cell_compartments.append(["axon"])
        i = 0
        for connected_cell in connection._to_cell_types:
            type = assert_attr(connected_cell, "type", node_name + ".{}".format(i))
            i += 1
            if type not in self.cell_types:
                raise ConfigurationError(
                    "Unknown cell type '{}' in '{}.to_cell_types'".format(type, node_name)
                )
            to_cell_types.append(self.cell_types[type])
            if "compartments" in connected_cell:
                to_cell_compartments.append(connected_cell["compartments"])
            else:
                to_cell_compartments.append(["dendrites"])
        connection.__dict__["from_cell_types"] = from_cell_types
        connection.__dict__["to_cell_types"] = to_cell_types
        connection.__dict__["from_cell_compartments"] = from_cell_compartments
        connection.__dict__["to_cell_compartments"] = to_cell_compartments

    def init_simulation_component(self, name, section, component_class, adapter):
        component = load_configurable_class(
            name, component_class, SimulationComponent, parameters={"adapter": adapter}
        )
        fill_configurable_class(component, section)
        return component


class PlottingConfig:
    def __init__(self, name, color="#000000", opacity=1.0):
        self.label = name
        self.color = color
        self.opacity = opacity
