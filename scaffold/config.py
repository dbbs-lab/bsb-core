import os
import configparser
from .models import CellType, Layer, GeometricCellType, MorphologicCellType
from .quantities import parseToMicrometer, parseToDensity
from .geometries import Geometry as BaseGeometry
from .helpers import copyIniKey
from pprint import pprint

class ScaffoldConfig(object):

    def __init__(self):
        # Initialise empty config object.

        # Dictionaries and lists
        self.CellTypes = {}
        self.CellTypeIDs = []
        self.Layers = {}
        self.LayerIDs = []
        self.Connections = {}
        self.Geometries = {}

        # General simulation values
        self.X = 200 * 10 ** -6    # Transverse dimension size (m)
        self.Z = 200 * 10 ** -6    # Longitudinal dimension size (m)

    def addCellType(self, cellType):
        '''
            Adds a cell type to the config object. Cell types are used to populate
            cells into the layers of the simulation.

            :param cellType: CellType object to add
            :type cellType: CellType
        '''
        # Register a new cell type model.
        self.CellTypes[cellType.name] = cellType
        self.CellTypeIDs.append(cellType.name)

    def addGeometry(self, geometry):
        '''
            Adds a geometry to the config object. Geometries are used to determine
            which cells touch and form synapses.

            :param geometry: Geometry object to add
            :type geometry: Geometry
        '''
        # Register a new geometry model.
        self.Geometries[geometry.name] = geometry

    def addLayer(self, layer):
        '''
            Adds a layer to the config object. Layers are regions of the simulation
            to be populated by cells.

            :param layer: Layer object to add
            :type layer: Layer
        '''
        # Register a new layer model.
        self.Layers[layer.name] = layer
        self.LayerIDs.append(layer.name)

    def getLayer(self, name='',id=-1):
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
            if len(self.LayerIDs) <= id:
                raise Exception("Layer with id {} not found.".format(id))
            return list(self.Layers.values())[id]
        if name != '':
            if not name in self.Layers:
                raise Exception("Layer with name '{}' not found".format(name))
            return self.Layers[name]
        raise Exception("Invalid arguments for ScaffoldConfig.getLayer: name='{}', id={}".format(name, id))

    def getLayerID(self, name):
        return self.LayerIDs.index(name)

    def getLayerList(self):
        return list(self.Layers.values())

class ScaffoldIniConfig(ScaffoldConfig):
    '''
        Create a scaffold configuration from a .ini file.
    '''

    def __init__(self, file):
        '''
            Initialize ScaffoldIniConfig from .ini file.

            :param file: Path of the configuration .ini file.
            :type file: string
        '''

        # Initialize base config class
        ScaffoldConfig.__init__(self)
        # Determine extension of file.
        head, tail = os.path.splitext(file)
        # Append .ini and send warning if .ini extension is not present.
        if tail != '.ini':
            print("[WARNING] No .ini extension on given config file '{}', config file changed to : '{}'".format(file, file + '.ini'))
            file = file + '.ini'
        # Use configparser to read .ini file
        parsedConfig = configparser.ConfigParser()
        parsedConfig.read(file)
        self._sections = parsedConfig.sections()
        self._config = parsedConfig
        # Check if ini file is empty
        if len(self._sections) == 0:
            raise Exception("Empty or non existent configuration file '{}'.".format(file))
        # Define a map from section types to initializers.
        sectionInitializers = {
            'Cell': self.iniCellType,
            'Layer': self.iniLayer,
            'Geometry': self.iniGeometry,
        }
        # Define a map from section types to finalizers.
        sectionFinalizers = {
            'Cell': self.finalizeCellType,
            'Layer': self.finalizeLayer,
            'Geometry': self.finalizeGeometry,
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
            sectionFinalizers[sectionType](sectionConfig)

    def iniCellType(self, name, section):
        '''
            Initialise a CellType from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`CellType`: object.
            :rtype: CellType
        '''
        # Morphology type
        if not 'morphologytype' in section:
            raise Exception('Required attribute MorphologyType missing in {} section.'.format(name))
        morphoType = section['morphologytype']
        # Construct geometrical/morphological cell type.
        if morphoType == 'Geometry':
            cellType = self.iniGeometricCell(name, section)
        elif morphoType == 'Morphology':
            cellType = self.iniMorphologicCell(name, section)
        else:
            raise Exception("Cell morphology type must be either 'Geometry' or 'Morphology'")
        # Radius
        if not 'radius' in section:
            raise Exception('Required attribute Radius missing in {} section.'.format(name))
        cellType.radius = float(section['radius'])
        # Density
        if not 'density' in section and (not 'ratio' in section or not 'ratioto' in section):
            raise Exception('Either Density or Ratio and RatioTo attributes missing in {} section.'.format(name))
        if 'density' in section:
            cellType.density = parseToDensity(section['density'])
        else:
            cellType.ratio = float(section['ratio'])
            cellType.ratioTo = section['ratioTo']
        # Color
        if 'color' in section:
            cellType.color = section['color']
        # Register cell type
        self.addCellType(cellType)
        return cellType

    def iniGeometricCell(self, name, section):
        '''
            Create a cell type that is modelled in space based on geometrical rules.
        '''
        # Geometry name
        if not 'geometryname' in section:
            raise Exception('Required geometry attribute GeometryName missing in {} section.'.format(name))
        geometryName = section['geometryname']
        if not geometryName in self.Geometries.keys():
            raise Exception("Unknown geometry '{}' in section '{}'".format(geometryName, name))

        # Create and return geometric cell type
        cellType = GeometricCellType(name, self.Geometries[geometryName])
        return cellType

    def iniMorphologicCell(self, name, section):
        '''
            Create a cell type that is modelled in space based on a detailed morphology.
        '''
        return MorphologicCellType(name)

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

        thickness = parseToMicrometer(section['thickness'])
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
            layers = self.getLayerList()
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
        #Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        # Add layer to the config object
        self.addLayer(layer)
        return layer

    def iniGeometry(self, name, section):
        '''
            Initialize a Geometry-subclass from the configuration. Uses __import__
            to fetch geometry class, then copies all keys as is from config section to instance
            and adds it to the Geometries dictionary.
        '''
        # Keys to exclude from copying to the geometry instance
        excluded = ['MorphologyType', 'GeometryName']
        if not 'class' in section:
            raise Exception('Required attribute Class missing in {} section.'.format(name))
        classParts = section['class'].split('.')
        className = classParts[-1]
        moduleName = '.'.join(classParts[:-1])
        moduleRef = __import__(moduleName, globals(), locals(), [className], 0)
        classRef = moduleRef.__dict__[className]
        if not issubclass(classRef, BaseGeometry):
            raise Exception("Class '{}.{}' must derive from scaffold.geometries.Geometry".format(moduleName,className))
        geometryInstance = classRef()
        for key in section:
            if not key in excluded:
                copyIniKey(geometryInstance, section, {'key': key, 'type': 'string'})
        geometryInstance.__dict__['name'] = name
        self.addGeometry(geometryInstance)


    def finalizeGeometry(self, section):
        pass

    def finalizeLayer(self, section):
        pass

    def finalizeCellType(self, section):
        # TODO: Load morphology/geometry config
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
