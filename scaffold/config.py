import os
import configparser
from .models import CellType, Layer

class ScaffoldConfig(object):

    def __init__(self):
        # Initialise empty config object.

        # Dictionaries and lists
        self.CellTypes = {}
        self.CellTypeIDs = []
        self.Layers = {}
        self.LayerIDs = []
        self.Connections = {}

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
        if tail != 'ini':
            print("[WARNING] No .ini extension on given config file '{}', config file changed to : '{}'".format(file, file + '.ini'))
            file = file + '.ini'
        # Use configparser to read .ini file
        cp = configparser.ConfigParser()
        cp.read(file)
        self._sections = cp.sections()
        self._config = cp
        # Check if ini file is empty
        if len(self._sections) == 0:
            raise Exception("Empty or non existent configuration file '{}'.".format(file))
        # Define a map from section types to initializers.
        sectionInitializers = {
            'Cell': self.iniCellType,
            'Layer': self.iniLayer
        }
        # Define a map from section types to finalizers.
        sectionFinalizers = {
            'Cell': self.finalizeCellType,
            'Layer': self.finalizeLayer
        }
        # Initialize special sections such as the general section.
        self.initSections()
        # Initialize each section in the .ini file based on their type
        for sectionName in self._sections:
            sectionConfig = cp[sectionName]
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
            sectionInitializers[sectionType](sectionConfig)
        # Finalize each section in the .ini file based on their type.
        # Finalisation allows sections to configure themselves based on properties initialised in other sections.
        for sectionName in self._sections:
            sectionConfig = cp[sectionName]
            sectionType = sectionConfig['type']
            sectionFinalizers[sectionType](sectionConfig)

    def iniCellType(self, section):
        name = section['name']
        self.addCellType(CellType(name))

    def iniLayer(self, section):
        '''
            Initialise a Layer from a .ini object.

            :param section: A section of a .ini file, parsed by configparser.
            :type section: /

            :returns: A :class:`Layer`: object.
            :rtype: Layer
        '''
        name = section['name']
        # Set thickness of the layer
        if not 'thickness' in section:
            raise Exception('Required attribute Thickness missing in {} section.'.format(name))
        thickness = float(section['thickness']) * 10 ** -6
        # Set the distance between the bottom of this layer and the bottom of the simulation.
        if not 'position' in section:
            origin = [0., 0., 0.]
        else:
            origin = [float(coord) for coord in section['position'].split(',')]
            if len(origin) != 3:
                raise Exception("Invalid position '{}' given in section '{}'".format(section['position'], name))
        # Stack this layer on the previous one.
        if 'stack' in section and section['stack'] != 'False':
            # List the layers
            layers = list(self.Layers.values())
            if len(layers) == 0:
                # If this is the first layer, put it at the bottom of the simulation.
                origin[1] = 0.
            else:
                # Otherwise, place it on top of the previous one
                previousLayer = layers[-1]
                origin[1] = previousLayer.origin[1] + previousLayer.dimensions[1]
        # Set the layer dimensions, first get the XZ-scaling factor, if present
        xzScale = 1.
        if 'xz-scale' in section:
            xzScale = float(section['xz-scale'])
        dimensions = [self.X * xzScale, thickness, self.Z * xzScale]
        #Put together the layer object from the extracted values.
        layer = Layer(name, origin, dimensions)
        print("adding layer {} at X {} Y {} Z {} size: {}, {}, {}".format(name, *origin, *dimensions))
        # Add layer to the config object
        self.addLayer(layer)

    def finalizeLayer(self, section):
        pass

    def finalizeCellType(self, section):
        pass

    def initSections(self):
        '''
            Initialize the special sections of the configuration file.
            Special sections: 'General'
        '''
        def copyIniToSelf(self, section, key_config):
            ini_key = key_config['key']
            if not ini_key in section: # Only copy values that exist in the config
                return

            def micrometer(value):
                return float(value) * 10 ** -6

            # Process the config values based on the type in their key_config.
            morph_map = {'micrometer': micrometer}
            self.__dict__[ini_key] = morph_map[key_config['type']](section[ini_key])

        special = ['General']

        # An array of keys to extract from the General section.
        # TODO: Quantulum. see #2
        general_keys = [
            {'key': 'X', 'type': 'micrometer'},
            {'key': 'Z', 'type': 'micrometer'}
        ]
        # Copy all general_keys from the General section to the config object.
        if 'General' in self._sections:
            for key in general_keys:
                copyIniToSelf(self, self._config['General'], key)

        # Filter out all special sections
        self._sections = list(filter(lambda x: not x in special, self._sections))
