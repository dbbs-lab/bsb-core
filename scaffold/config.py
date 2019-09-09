import os, abc
import configparser
from .models import CellType, Layer
from .geometries import Geometry as BaseGeometry
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
        self.geometries = {}
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
                print("[WARNING] No .ini extension on given config file '{}', config file changed to : '{}'".format(file, file + '.ini'))
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

    def addGeometry(self, geometry):
        '''
            Adds a geometry to the config object. Geometries are used to determine
            which cells touch and form synapses.

            :param geometry: Geometry object to add
            :type geometry: Geometry
        '''
        # Register a new Geometry.
        self.geometries[geometry.name] = geometry

    def addPlacementStrategy(self, placement):
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

    def get_layerList(self):
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

class ScaffoldIniConfig(ScaffoldConfig):
    '''
        Create a scaffold configuration from an INI formatted file/string.
    '''

    def __init__(self, file=None, stream=None, verbosity=0):
        '''
            Initialize ScaffoldIniConfig from .ini file.

            :param file: Path of the configuration .ini file.
            :type file: string
            :param stream: INI formatted string representing the configuration file.
            :type file: string
            :param verbosity: Verbosity (output level) of the scaffold.
            :type file: int
        '''
        # Set _extension to indicate we expect a .ini file.
        self._extension = ".ini"
        # Initialize base config class, handling the reading of file/stream to string
        ScaffoldConfig.__init__(self, file, stream, verbosity)
        # Use configparser to read self._raw ini string provided by parent ScaffoldConfig.__init__
        parsedConfig = configparser.ConfigParser()
        parsedConfig.read_string(self._raw)
        self._sections = parsedConfig.sections()
        self._config = parsedConfig
        # Check if ini file is empty
        if len(self._sections) == 0:
            raise Exception("Empty or non existent configuration " + (("file" + '\'{}\''.format(file)) if file else "stream") + ".")
        # Defines a map from section types to initializers.
        sectionInitializers = {
            'Cell': self.iniCellType,
            'Layer': self.iniLayer,
            'Geometry': self.iniGeometry,
            'Connection': self.iniConnection,
            'Placement': self.iniPlacement,
        }
        # Defines a map from section types to finalizers.
        sectionFinalizers = {
            'Cell': self.finalizeCellType,
            'Layer': self.finalizeLayer,
            'Geometry': self.finalizeGeometry,
            'Connection': self.finalizeConnection,
            'Placement': self.finalizePlacement,
        }
        # Defines a map from section types to config object dictionaries
        sectionDictionaries = {
            'Cell': self.cell_types,
            'Layer': self.layers,
            'Geometry': self.geometries,
            'Connection': self.connection_types,
            'Placement': self.placement_strategies,
        }
        # Initialize special sections such as the general section.
        self.initSections()
        # Initialize each section in the .ini file based on their type
        for sectionName in self._sections:
            sectionConfig = parsedConfig[sectionName]
            if not 'type' in sectionConfig:
                raise Exception("No type declared for section '{}' in '{}'.".format(
                    sectionName,
                    file,
                ))
            sectionType = sectionConfig['type']
            if not sectionType in sectionInitializers:
                raise Exception("Unknown section type '{}' for section '{}' in '{}'. Options: '{}'".format(
                    sectionType,
                    sectionName,
                    file,
                    "', '".join(sectionInitializers.keys()) # Format a list of the available section types.
                ))
            # Call the appropriate ini-section initialiser for this type.
            sectionInitializers[sectionType](sectionName, sectionConfig)
        # Finalize each section in the .ini file based on their type.
        # Finalisation allows sections to configure themselves based on properties initialised in other sections.
        for sectionName in self._sections:
            sectionConfig = parsedConfig[sectionName]
            sectionType = sectionConfig['type']
            # Fetch the initialized config from the storage dictionaries and finalize it.
            sectionFinalizers[sectionType](sectionDictionaries[sectionType][sectionName], sectionConfig)

    def iniCellType(self, name, section):
        '''
            Initialise a CellType from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`CellType`: object.
            :rtype: CellType
        '''
        cell_type = CellType(name)
        # Radius
        if not 'radius' in section:
            raise Exception('Required attribute Radius missing in {} section.'.format(name))
        cell_type.radius = float(section['radius'])
        # Density
        if not 'density' in section and not 'planardensity' in section and (not 'ratio' in section or not 'ratioto' in section):
            raise Exception('Either Density, PlanarDensity or Ratio and RatioTo attributes missing in {} section.'.format(name))
        if 'density' in section:
            cell_type.density = float(section['density'])
        elif 'planardensity' in section:
            cell_type.planarDensity = float(section['planardensity'])
        else:
            cell_type.ratio = float(section['ratio'])
            cell_type.ratioTo = section['ratioTo']
        # Color
        if 'color' in section:
            cell_type.color = section['color']
        # Register cell type
        self.addCellType(cell_type)
        return cell_type

    def iniGeometricCell(self, cell_type, section):
        '''
            Create a cell type that is modelled in space based on geometrical rules.
        '''
        if not 'geometryname' in section:
            raise Exception('Required geometry attribute GeometryName missing in {} section.'.format(name))
        geometry_name = section['geometryname']
        if not geometry_name in self.geometries.keys():
            raise Exception("Unknown geometry '{}' in section '{}'".format(geometry_name, name))
        # Set the cell's geometry
        cell_type.set_geometry(self.geometries[geometry_name])
        return cell_type

    def iniMorphologicCell(self, cell_type, section):
        '''
            Create a cell type that is modelled in space based on a detailed morphology.
        '''
        raise Exception("Morphologic cells not implemented yet.")
        return cell_type

    def iniLayer(self, name, section):
        '''
            Initialise a Layer from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        '''
        # Get thickness of the layer
        if not 'thickness' in section:
            raise Exception('Required attribute Thickness missing in {} section.'.format(name))

        thickness = float(section['thickness'])
        # Set the position of this layer in the space.
        if not 'position' in section:
            origin = [0., 0., 0.]
        else:
            # TODO: Catch possible casting errors from string to float.
            origin = [float(coord) for coord in section['position'].split(',')]
            if len(origin) != 3:
                raise Exception("Invalid position '{}' given in section '{}'".format(section['position'], name))

        # Stack this layer on the previous one.
        if 'stack' in section and section['stack'] != 'False':
            layers = self.get_layerList()
            if len(layers) == 0:
                # If this is the first layer, put it at the bottom of the simulation.
                origin[1] = 0.
            else:
                # Otherwise, place it on top of the previous one
                previousLayer = layers[-1]
                origin[1] = previousLayer.origin[1] + previousLayer.dimensions[1]
        # Set the layer dimensions
        #   scale by the XZ-scaling factor, if present
        xzScale = 1.
        if 'xzscale' in section:
            xzScale = float(section['xzscale'])
        dimensions = [self.X * xzScale, thickness, self.Z * xzScale]
        # Center the layer on the XZ plane
        if 'xzcenter' in section and section['xzcenter'] == 'True':
            origin[0] = (self.X - dimensions[0]) / 2.
            origin[2] = (self.Z - dimensions[2]) / 2.
        # Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        # Add layer to the config object
        self.add_layer(layer)
        return layer

    def iniGeometry(self, name, section):
        '''
            Initialize a Geometry-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        # Keys to exclude from copying to the geometry instance
        excluded = ['Type', 'MorphologyType', 'GeometryName', 'Class']
        geometryInstance = self.loadConfigClass(name, section, BaseGeometry, excluded)
        self.addGeometry(geometryInstance)

    def iniConnection(self, name, section):
        '''
            Initialize a ConnectionStrategy-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        connectionInstance = self.loadConfigClass(name, section, ConnectionStrategy)
        self.addConnection(connectionInstance)

    def iniPlacement(self, name, section):
        '''
            Initialize a PlacementStrategy-subclass from the configuration. Uses __import__
            to fetch placement class, then copies all keys as is from config section to instance
            and adds it to the PlacementStrategies dictionary.
        '''
        # Keys to exclude from copying to the geometry instance
        placementInstance = self.loadConfigClass(name, section, PlacementStrategy)
        self.addPlacementStrategy(placementInstance)


    def finalizeGeometry(self, geometry, section):
        pass

    def finalizeLayer(self, layer, section):
        pass

    def finalizeCellType(self, cell_type, section):
        '''
            Adds configured morphology and placement strategy to the cell type configuration.
        '''

        # Morphology type
        if not 'morphologytype' in section:
            raise Exception('Required attribute MorphologyType missing in {} section.'.format(cell_type.name))
        morphoType = section['morphologytype']
        # Construct geometrical/morphological cell type.
        if morphoType == 'Geometry':
            self.iniGeometricCell(cell_type, section)
        elif morphoType == 'Morphology':
            self.iniMorphologicCell(cell_type, section)
        else:
            raise Exception("Cell morphology type must be either 'Geometry' or 'Morphology'")
        # Placement strategy
        if not 'placementstrategy' in section:
            raise Exception('Required attribute PlacementStrategy missing in {} section.'.format(cell_type.name))
        placementName = section['placementstrategy']
        if not placementName in self.placement_strategies:
            raise Exception("Unknown placement strategy '{}' in {} section".format(placementName, cell_type.name))
        if not cell_type.ratio is None:
            if cell_type.ratioTo not in self.cell_types:
                raise Exception("Ratio defined to unknown cell type '{}' in {}".format(cell_type.ratioTo, cell_type.name))
        cell_type.setPlacementStrategy(self.placement_strategies[placementName])

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

    def loadConfigClass(self, name, section, parentClass, excludedKeys = ['Type', 'Class']):
        if not 'class' in section:
            raise Exception('Required attribute Class missing in {} section.'.format(name))
        classParts = section['class'].split('.')
        className = classParts[-1]
        moduleName = '.'.join(classParts[:-1])
        moduleRef = __import__(moduleName, globals(), locals(), [className], 0)
        classRef = moduleRef.__dict__[className]
        if not issubclass(classRef, parentClass):
            raise Exception("Class '{}.{}' must derive from {}.{}".format(
                moduleName,
                className,
                parentClass.__module__,
                parentClass.__qualname__,
            ))
        instance = classRef()
        for key in section:
            if not key in excludedKeys:
                copyIniKey(instance, section, {'key': key, 'type': 'string'})
        instance.__dict__['name'] = name
        return instance


class ScaffoldJSONConfig(ScaffoldConfig):
    '''
        Create a scaffold configuration from a JSON formatted file/string.
    '''

    def __init__(self, file=None, stream=None, verbosity=0):
        '''
            Initialize ScaffoldIniConfig from .ini file.

            :param file: Path of the configuration .ini file.
            :type file: string
            :param stream: INI formatted string representing the configuration file.
            :type file: string
            :param verbosity: Verbosity (output level) of the scaffold.
            :type file: int
        '''
        # Set _extension to indicate we expect a .ini file.
        self._extension = ".ini"
        # Initialize base config class, handling the reading of file/stream to string
        ScaffoldConfig.__init__(self, file, stream, verbosity)
        # Use configparser to read self._raw ini string provided by parent ScaffoldConfig.__init__
        parsedConfig = configparser.ConfigParser()
        parsedConfig.read_string(self._raw)
        self._sections = parsedConfig.sections()
        self._config = parsedConfig
        # Check if ini file is empty
        if len(self._sections) == 0:
            raise Exception("Empty or non existent configuration " + (("file" + '\'{}\''.format(file)) if file else "stream") + ".")
        # Defines a map from section types to initializers.
        sectionInitializers = {
            'Cell': self.iniCellType,
            'Layer': self.iniLayer,
            'Geometry': self.iniGeometry,
            'Connection': self.iniConnection,
            'Placement': self.iniPlacement,
        }
        # Defines a map from section types to finalizers.
        sectionFinalizers = {
            'Cell': self.finalizeCellType,
            'Layer': self.finalizeLayer,
            'Geometry': self.finalizeGeometry,
            'Connection': self.finalizeConnection,
            'Placement': self.finalizePlacement,
        }
        # Defines a map from section types to config object dictionaries
        sectionDictionaries = {
            'Cell': self.cell_types,
            'Layer': self.layers,
            'Geometry': self.geometries,
            'Connection': self.connection_types,
            'Placement': self.placement_strategies,
        }
        # Initialize special sections such as the general section.
        self.initSections()
        # Initialize each section in the .ini file based on their type
        for sectionName in self._sections:
            sectionConfig = parsedConfig[sectionName]
            if not 'type' in sectionConfig:
                raise Exception("No type declared for section '{}' in '{}'.".format(
                    sectionName,
                    file,
                ))
            sectionType = sectionConfig['type']
            if not sectionType in sectionInitializers:
                raise Exception("Unknown section type '{}' for section '{}' in '{}'. Options: '{}'".format(
                    sectionType,
                    sectionName,
                    file,
                    "', '".join(sectionInitializers.keys()) # Format a list of the available section types.
                ))
            # Call the appropriate ini-section initialiser for this type.
            sectionInitializers[sectionType](sectionName, sectionConfig)
        # Finalize each section in the .ini file based on their type.
        # Finalisation allows sections to configure themselves based on properties initialised in other sections.
        for sectionName in self._sections:
            sectionConfig = parsedConfig[sectionName]
            sectionType = sectionConfig['type']
            # Fetch the initialized config from the storage dictionaries and finalize it.
            sectionFinalizers[sectionType](sectionDictionaries[sectionType][sectionName], sectionConfig)

    def iniCellType(self, name, section):
        '''
            Initialise a CellType from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`CellType`: object.
            :rtype: CellType
        '''
        cell_type = CellType(name)
        # Radius
        if not 'radius' in section:
            raise Exception('Required attribute Radius missing in {} section.'.format(name))
        cell_type.radius = float(section['radius'])
        # Density
        if not 'density' in section and not 'planardensity' in section and (not 'ratio' in section or not 'ratioto' in section):
            raise Exception('Either Density, PlanarDensity or Ratio and RatioTo attributes missing in {} section.'.format(name))
        if 'density' in section:
            cell_type.density = float(section['density'])
        elif 'planardensity' in section:
            cell_type.planarDensity = float(section['planardensity'])
        else:
            cell_type.ratio = float(section['ratio'])
            cell_type.ratioTo = section['ratioTo']
        # Color
        if 'color' in section:
            cell_type.color = section['color']
        # Register cell type
        self.addCellType(cell_type)
        return cell_type

    def iniGeometricCell(self, cell_type, section):
        '''
            Create a cell type that is modelled in space based on geometrical rules.
        '''
        if not 'geometryname' in section:
            raise Exception('Required geometry attribute GeometryName missing in {} section.'.format(name))
        geometry_name = section['geometryname']
        if not geometry_name in self.geometries.keys():
            raise Exception("Unknown geometry '{}' in section '{}'".format(geometry_name, name))
        # Set the cell's geometry
        cell_type.set_geometry(self.geometries[geometry_name])
        return cell_type

    def iniMorphologicCell(self, cell_type, section):
        '''
            Create a cell type that is modelled in space based on a detailed morphology.
        '''
        raise Exception("Morphologic cells not implemented yet.")
        return cell_type

    def iniLayer(self, name, section):
        '''
            Initialise a Layer from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        '''
        # Get thickness of the layer
        if not 'thickness' in section:
            raise Exception('Required attribute Thickness missing in {} section.'.format(name))

        thickness = float(section['thickness'])
        # Set the position of this layer in the space.
        if not 'position' in section:
            origin = [0., 0., 0.]
        else:
            # TODO: Catch possible casting errors from string to float.
            origin = [float(coord) for coord in section['position'].split(',')]
            if len(origin) != 3:
                raise Exception("Invalid position '{}' given in section '{}'".format(section['position'], name))

        # Stack this layer on the previous one.
        if 'stack' in section and section['stack'] != 'False':
            layers = self.get_layerList()
            if len(layers) == 0:
                # If this is the first layer, put it at the bottom of the simulation.
                origin[1] = 0.
            else:
                # Otherwise, place it on top of the previous one
                previousLayer = layers[-1]
                origin[1] = previousLayer.origin[1] + previousLayer.dimensions[1]
        # Set the layer dimensions
        #   scale by the XZ-scaling factor, if present
        xzScale = 1.
        if 'xzscale' in section:
            xzScale = float(section['xzscale'])
        dimensions = [self.X * xzScale, thickness, self.Z * xzScale]
        # Center the layer on the XZ plane
        if 'xzcenter' in section and section['xzcenter'] == 'True':
            origin[0] = (self.X - dimensions[0]) / 2.
            origin[2] = (self.Z - dimensions[2]) / 2.
        # Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        # Add layer to the config object
        self.add_layer(layer)
        return layer

    def iniGeometry(self, name, section):
        '''
            Initialize a Geometry-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        # Keys to exclude from copying to the geometry instance
        excluded = ['Type', 'MorphologyType', 'GeometryName', 'Class']
        geometryInstance = self.loadConfigClass(name, section, BaseGeometry, excluded)
        self.addGeometry(geometryInstance)

    def iniConnection(self, name, section):
        '''
            Initialize a ConnectionStrategy-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        connectionInstance = self.loadConfigClass(name, section, ConnectionStrategy)
        self.addConnection(connectionInstance)

    def iniPlacement(self, name, section):
        '''
            Initialize a PlacementStrategy-subclass from the configuration. Uses __import__
            to fetch placement class, then copies all keys as is from config section to instance
            and adds it to the PlacementStrategies dictionary.
        '''
        # Keys to exclude from copying to the geometry instance
        placementInstance = self.loadConfigClass(name, section, PlacementStrategy)
        self.addPlacementStrategy(placementInstance)


    def finalizeGeometry(self, geometry, section):
        pass

    def finalizeLayer(self, layer, section):
        pass

    def finalizeCellType(self, cell_type, section):
        '''
            Adds configured morphology and placement strategy to the cell type configuration.
        '''

        # Morphology type
        if not 'morphologytype' in section:
            raise Exception('Required attribute MorphologyType missing in {} section.'.format(cell_type.name))
        morphoType = section['morphologytype']
        # Construct geometrical/morphological cell type.
        if morphoType == 'Geometry':
            self.iniGeometricCell(cell_type, section)
        elif morphoType == 'Morphology':
            self.iniMorphologicCell(cell_type, section)
        else:
            raise Exception("Cell morphology type must be either 'Geometry' or 'Morphology'")
        # Placement strategy
        if not 'placementstrategy' in section:
            raise Exception('Required attribute PlacementStrategy missing in {} section.'.format(cell_type.name))
        placementName = section['placementstrategy']
        if not placementName in self.placement_strategies:
            raise Exception("Unknown placement strategy '{}' in {} section".format(placementName, cell_type.name))
        if not cell_type.ratio is None:
            if cell_type.ratioTo not in self.cell_types:
                raise Exception("Ratio defined to unknown cell type '{}' in {}".format(cell_type.ratioTo, cell_type.name))
        cell_type.setPlacementStrategy(self.placement_strategies[placementName])

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

    def loadConfigClass(self, name, section, parentClass, excludedKeys = ['Type', 'Class']):
        if not 'class' in section:
            raise Exception('Required attribute Class missing in {} section.'.format(name))
        classParts = section['class'].split('.')
        className = classParts[-1]
        moduleName = '.'.join(classParts[:-1])
        moduleRef = __import__(moduleName, globals(), locals(), [className], 0)
        classRef = moduleRef.__dict__[className]
        if not issubclass(classRef, parentClass):
            raise Exception("Class '{}.{}' must derive from {}.{}".format(
                moduleName,
                className,
                parentClass.__module__,
                parentClass.__qualname__,
            ))
        instance = classRef()
        for key in section:
            if not key in excludedKeys:
                copyIniKey(instance, section, {'key': key, 'type': 'string'})
        instance.__dict__['name'] = name
        return instance
