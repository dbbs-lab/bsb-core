import os, abc
from .models import CellType, Layer
from .morphologies import Morphology as BaseMorphology
from .connectivity import ConnectionStrategy
from .placement import PlacementStrategy
from .helpers import copyIniKey

class ScaffoldConfig(object):

    def __init__(self, file=None, stream=None, verbosity=0):
        # Initialise empty config object.

        # Dictionaries and lists
        self.cell_types = {}
        self.cell_type_map = []
        self.layers = {}
        self.layer_map = []
        self.connection_types = {}
        self.morphologies = {}
        self.placement_strategies = {}
        self.verbosity = verbosity
        self._raw = ''
        if not hasattr(self, '_extension'):
            self._extension = ''

        # Fallback simulation values
        self.X = 200    # Transverse simulation space size (µm)
        self.Z = 200    # Longitudinal simulation space size (µm)

        self.read_config(file, stream)

    def read_config(self, file=None, stream=None):
        if not stream is None:
            self.read_config_stream(stream)
        elif not file is None:
            self.read_config_file(file)

    def read_config_file(self, file):
        # Determine extension of file.
        head, tail = os.path.splitext(file)
        # Append .ini and send warning if .ini extension is not present.
        if tail != self._extension:
            if self.verbosity > 0:
                print("[WARNING] No {} extension on given config file '{}', config file changed to : '{}'".format(
                    self._extension,
                    file,
                    file + self._extension
                ))
            file += self._extension
        try:
            with open('scaffold/configurations/' + file, 'r') as file:
                self._raw = file.read()
        except Exception as e:
            with open(file, 'r') as file:
                self._raw = file.read()

    def read_config_stream(self, stream):
        self._raw = stream

    def addCellType(self, cell_type):
        '''
            Adds a cell type to the config object. Cell types are used to populate
            cells into the layers of the simulation.

            :param cell_type: CellType object to add
            :type cell_type: CellType
        '''
        # Register a new cell type model.
        self.cell_types[cell_type.name] = cell_type
        self.cell_type_map.append(cell_type.name)

    def addMorphology(self, morphology):
        '''
            Adds a morphology to the config object. Mrophologies are used to determine
            which cells touch and form synapses.

            :param morphology: Morphology object to add
            :type morphology: Morphology
        '''
        # Register a new Geometry.
        self.morphologies[morphology.name] = morphology

    def add_placement_strategy(self, placement):
        '''
            Adds a placement to the config object. Placement strategies are used to
            place cells in the simulation volume.

            :param placement: PlacementStrategy object to add
            :type placement: PlacementStrategy
        '''
        # Register a new Geometry.
        self.placement_strategies[placement.name] = placement

    def addConnection(self, connection):
        '''
            Adds a ConnectionStrategy to the config object. ConnectionStrategies
            are used to determine which touching cells to connect.

            :param connection: ConnectionStrategy object to add
            :type connection: ConnectionStrategy
        '''
        # Register a new ConnectionStrategy.
        self.connection_types[connection.name] = connection

    def add_layer(self, layer):
        '''
            Adds a layer to the config object. layers are regions of the simulation
            to be populated by cells.

            :param layer: Layer object to add
            :type layer: Layer
        '''
        # Register a new layer model.
        self.layers[layer.name] = layer
        self.layer_map.append(layer.name)

    def get_layer(self, name='',id=-1):
        '''
            Finds a layer by its name or id.

            :param name: Name of the layer to look for.
            :type name: string
            :param id: Id of the layer to look for.
            :type id: int

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        '''
        if id > -1:
            if len(self.layer_map) <= id:
                raise Exception("Layer with id {} not found.".format(id))
            return list(self.layers.values())[id]
        if name != '':
            if not name in self.layers:
                raise Exception("Layer with name '{}' not found".format(name))
            return self.layers[name]
        raise Exception("Invalid arguments for ScaffoldConfig.get_layer: name='{}', id={}".format(name, id))

    def get_layerID(self, name):
        return self.layer_map.index(name)

    def get_layer_list(self):
        return list(self.layers.values())

    def resize(self, X=None, Z=None):
        scaling_x = 1.
        scaling_z = 1.
        if not X is None:
            scaling_x = X / self.X
            self.X = X
        if not Z is None:
            scaling_z = Z / self.Z
            self.Z = Z
        for layer in self.layers.values():
            if layer.scaling:
                layer.dimensions[0] *= scaling_x
                layer.dimensions[2] *= scaling_z

    def load_configurable_class(self, name, configured_class_name, parent_class):
        classParts = configured_class_name.split('.')
        className = classParts[-1]
        moduleName = '.'.join(classParts[:-1])
        module_ref = __import__(moduleName, globals(), locals(), [className], 0)
        if not className in module_ref.__dict__:
            raise ConfigurableClassNotFoundException('Class not found:' + configured_class_name)
        class_ref = module_ref.__dict__[className]
        if not issubclass(class_ref, parent_class):
            raise Exception("Configurable class '{}.{}' must derive from {}.{}".format(
                moduleName,
                className,
                parent_class.__module__,
                parent_class.__qualname__,
            ))
        instance = class_ref()
        instance.__dict__['name'] = name
        return instance

    def fill_configurable_class(self, obj, conf, excluded=[]):
        for name, prop in conf.items():
            if not name in excluded:
                obj.__dict__[name] = prop

class JSONConfig(ScaffoldConfig):
    '''
        Create a scaffold configuration from a JSON formatted file/string.
    '''

    def __init__(self, file=None, stream=None, verbosity=0):
        '''
            Initialize config from .json file.

            :param file: Path of the configuration .json file.
            :type file: string
            :param stream: INI formatted string representing the configuration file.
            :type file: string
            :param verbosity: Verbosity (output level) of the scaffold.
            :type file: int
        '''
        import json
        # Set flags to indicate we expect a json configuration.
        self._type = 'json'
        self._extension = ".json"
        # Initialize base config class, handling the reading of file/stream to string
        ScaffoldConfig.__init__(self, file, stream, verbosity)
        # Use json module to read self._raw json string provided by parent ScaffoldConfig.__init__
        try:
            parsed_config = json.loads(self._raw)
        except json.decoder.JSONDecodeError as e:
            raise Exception("Error while loading JSON configuration: {}".format(e))

        self.load_general(parsed_config)
        self._layer_stacks = {}
        self.load_attr(config=parsed_config, attr='layers', init=self.init_layer, final=self.finalizeLayers, single=True)
        self.load_attr(config=parsed_config, attr='cell_types', init=self.init_cell_type, final=self.finalizeCellType)

    def load_general(self, config):
        '''
            Load the general segment in a JSON configuration file.
        '''
        if not 'network_architecture' in config:
            raise Exception("Missing 'network_architecture' attribute in configuration.")
        netw_config = config['network_architecture']
        if not 'simulation_volume_x' in netw_config:
            raise Exception("Missing 'simulation_volume_x' attribute in 'network_architecture' configuration.")
        if not 'simulation_volume_z' in netw_config:
            raise Exception("Missing 'simulation_volume_x' attribute in 'network_architecture' configuration.")
        self.X = float(netw_config['simulation_volume_x'])
        self.Z = float(netw_config['simulation_volume_z'])

    def load_attr(self, config, attr, init, final, single=False):
        '''
            Load an attribute of a config node containing a group of definitions .
        '''
        if not attr in config:
            raise Exception("Missing '{}' attribute in configuration.".format(attr))
        for def_name, def_config in config[attr].items():
            init(def_name, def_config)
        if single:
            final()
        else:
            for def_name, def_config in config[attr].items():
                final(def_name, def_config)

    def load_connection_types(self, config):
        pass

    def load_simulations(self, config):
        pass

    def init_cell_type(self, name, section):
        '''
            Initialise a CellType from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`CellType`: object.
            :rtype: CellType
        '''
        cell_type = CellType(name)
        node_name = 'cell_types.{}'.format(name)

        # Placement configuration
        placement_kwargs = {}
        # Get the placement configuration node
        placement = assert_attr(section, 'placement', node_name)
        cell_type.placement = self.init_placement(placement, name)
        # Get the morphology configuration node
        morphology = assert_attr(section, 'morphology', node_name)
        cell_type.morphology = self.init_morphology(morphology, name)
        cell_type.plotting = type('Plotting', (object,), {'color': '#000000'})()
        if 'plotting' in section:
            cell_type.plotting.color = if_attr(section['plotting'], 'color', '#000000')
        # Register cell type
        self.addCellType(cell_type)
        return cell_type

    def init_layer(self, name, config):
        '''
            Initialise a Layer from a json object.

            :param config: An object in the 'layers' array of the JSON file.
            :type config: /

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        '''
        # Get thickness of the layer
        if not 'thickness' in config:
            raise Exception('Required attribute thickness missing in {} config.'.format(name))
        thickness = float(config['thickness'])
        # Set the position of this layer in the space.
        if not 'position' in config:
            origin = [0., 0., 0.]
        else:
            # TODO: Catch possible casting errors to float.
            origin = [float(coord) for coord in config['position']]
            if len(origin) != 3:
                raise Exception("Invalid position '{}' given in config '{}'".format(config['position'], name))

        # Parse the layer stack config
        if 'stack' in config:
            stack_config = config['stack']
            if not 'stack_id' in stack_config:
                raise Exception("A 'stack_id' attribute is required in '{}.stack'.".format(name))
            stack_id = int(stack_config['stack_id'])
            stack = {'layers': {}}
            # Get or add stack from/to layer_stacks
            if stack_id in self._layer_stacks:
                stack = self._layer_stacks[stack_id]
            else:
                self._layer_stacks[stack_id] = stack
            if not 'position_in_stack' in stack_config:
                raise Exception("A 'position_in_stack' attribute is required in '{}.stack'.".format(name))
            stack['layers'][stack_config['position_in_stack']] = name
            # This config determines the position of the stack
            if 'position' in stack_config:
                if 'position' in stack:
                    raise Exception("Duplicate positioning attribute found for stack with id '{}'".format(stack_id))
                stack['position'] = stack_config['position']
        # Set the layer dimensions
        #   scale by the XZ-scaling factor, if present
        xzScale = 1.
        if 'xz_scale' in config:
            xzScale = float(config['xz_scale'])
        dimensions = [self.X * xzScale, thickness, self.Z * xzScale]
        #   and center the layer on the XZ plane, if present
        if 'xz_center' in config and config['xz_center'] == True:
            origin[0] = (self.X - dimensions[0]) / 2.
            origin[2] = (self.Z - dimensions[2]) / 2.
        # Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        # Add layer to the config object
        self.add_layer(layer)
        return layer

    def init_morphology(self, section, cell_type_name):
        '''
            Initialize a Geometry-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        name = cell_type_name + '_morphology'
        node_name = 'cell_types.{}.morphology'.format(cell_type_name)
        morphology_class = assert_attr(section, 'class', node_name)
        morphology = self.load_configurable_class(name, morphology_class, BaseMorphology)
        self.fill_configurable_class(morphology, section, excluded=['class'])
        self.addMorphology(morphology)
        return morphology

    def iniConnection(self, name, section):
        '''
            Initialize a ConnectionStrategy-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        connectionInstance = self.loadConfigClass(name, section, ConnectionStrategy)
        self.addConnection(connectionInstance)

    def init_placement(self, section, cell_type_name):
        '''
            Initialize a PlacementStrategy-subclass from the configuration. Uses __import__
            to fetch placement class, then copies all keys as is from config section to instance
            and adds it to the PlacementStrategies dictionary.
        '''
        name = cell_type_name + '_placement'
        node_name = 'cell_types.{}.placement'.format(cell_type_name)
        placement_class = assert_attr(section, 'class', node_name)
        try:
            placement = self.load_configurable_class(name, placement_class, PlacementStrategy)
        except ConfigurableClassNotFoundException as e:
            raise Exception("Couldn't find class '{}' specified in '{}'".format(placement_class, node_name))
        # Placement layer
        placement.layer = assert_attr(section, 'layer', node_name)
        # Radius of the cell soma
        placement.radius = assert_attr_float(section, 'soma_radius', node_name)
        # Density configurations all rely on a float or a float and relation
        density_attr, density_value = assert_strictly_one(section, ['density', 'planar_density', 'placement_count_ratio', 'density_ratio'], node_name)
        density_value = assert_float(density_value, '{}.{}'.format(node_name, density_attr))
        placement.__dict__[density_attr] = density_value
        # Does this density configuration rely on a relation to another celltype?
        ratio_attrs = ['placement_count_ratio', 'density_ratio']
        if density_attr in ratio_attrs:
            relation = assert_attr(section, 'placement_relative_to', node_name)
            placement.placement_relative_to = relation

        # Copy other information to be validated by the placement class
        self.fill_configurable_class(placement, section, excluded=['class', 'layer', 'soma_radius', 'density', 'planar_density', 'placement_count_ratio', 'density_ratio'])

        # Register the configured placement class
        self.add_placement_strategy(placement)
        return placement


    def finalizeGeometry(self, geometry, section):
        pass

    def finalizeLayers(self):
        for stack in self._layer_stacks.values():
            if not 'position' in stack:
                stack['position'] = [0., 0., 0.]
            # Get the current roof of the stack
            stack_roof = stack['position'][1]
            for (id, name) in sorted(stack["layers"].items()):
                layer = self.layers[name]
                # Place the layer on top of the roof of the stack, and move up the roof by the thickness of the stacked layer.
                layer.origin[1] = stack_roof
                stack_roof += layer.thickness

    def finalizeCellType(self, cell_type, section):
        '''
            Finalize configuration of the cell type.
        '''
        pass

    def finalizeConnection(self, connection, section):
        if not hasattr(connection, 'cellfrom'):
            raise Exception("Required attribute 'CellFrom' missing in {}".format(connection.name))
        if not hasattr(connection, 'cellto'):
            raise Exception("Required attribute 'CellTo' missing in {}".format(connection.name))
        if not connection.cellfrom in self.cell_types:
            raise Exception("Unknown CellFrom '{}' in {}".format(connection.cellfrom, connection.name))
        if not connection.cellto in self.cell_types:
            raise Exception("Unknown CellTo '{}' in {}".format(connection.cellto, connection.name))
        connection.__dict__['from_celltype'] = self.cell_types[connection.cellfrom]
        connection.__dict__['to_celltype'] = self.cell_types[connection.cellto]

    def finalizePlacement(self, placement, section):
        pass

    def initSections(self):
        '''
            Initialize the special sections of the configuration file.
            Special sections: 'General'
        '''
        special = ['General']
        # An array of keys to extract from the General section.
        general_keys = [
            {'key': 'X', 'type': 'micrometer'},
            {'key': 'Z', 'type': 'micrometer'}
        ]
        # Copy all general_keys from the General section to the config object.
        if 'General' in self._sections:
            for key in general_keys:
                copyIniKey(self, self._config['General'], key)

        # Filter out all special sections
        self._sections = list(filter(lambda x: not x in special, self._sections))

def assert_attr(section, attr, section_name):
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    return section[attr]

def if_attr(section, attr, default_value):
    if not attr in section:
        return default_value
    return section[attr]

def assert_strictly_one(section, attrs, section_name):
    attr_list = []
    for attr in attrs:
        if attr in section:
            attr_list.append(attr)
    if len(attr_list) != 1:
        msg = "{} found: ".format(len(attr_list)) + ", ".join(attr_list)
        if len(attr_list) == 0:
            msg = "None found."
        raise Exception("Strictly one of the following attributes is expected in {}: {}. {}".format(section_name, ", ".join(attrs), msg))
    else:
        return attr_list[0], section[attr_list[0]]

def assert_float(val, section_name):
    try:
        ret = float(val)
    except ValueError as e:
        raise Exception("Invalid float '{}' given for '{}'".format(val, section_name))
    return ret

def assert_attr_float(section, attr, section_name):
    if not attr in section:
        raise Exception("Required attribute '{}' missing in '{}'".format(attr, section_name))
    return assert_float(section[attr], section_name)

class ConfigurableClassNotFoundException(Exception):
    pass
