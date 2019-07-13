import os
import configparser
from .models import CellType, Layer

class ScaffoldConfig(object):

    def __init__(self):
        # Initialise empty config object.
        self.CellTypes = {}
        self.Layers = {}
        self.Connections = {}

    def addCellType(self, cellType):
        # Register a new cell type model.
        self.CellTypes[cellType.name] = cellType

    def addLayer(self, layer):
        # Register a new cell type model.
        self.Layers[layer.name] = layer

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
        # Check if ini file is empty
        if len(self._sections) == 0:
            raise Exception("Empty or non existent configuration file '{}'.".format(file))
        # Define a map from section types to initializers.
        sectionTypes = {
            'Cell': self.iniCellType,
            'Layer': self.iniLayer
        }
        # Initialize each section in the .ini file based on their type
        for sectionName in self._sections:
            sectionConfig = cp[sectionName]
            if not 'type' in sectionConfig:
                raise Exception("No type declared for section '{}' in '{}'.".format(
                    sectionName,
                    file,
                ))
            sectionType = sectionConfig['type']
            if not sectionType  in sectionTypes:
                raise Exception("Unknown section type '{}' for section '{}' in '{}'. Options: '{}'".format(
                    sectionType,
                    sectionName,
                    file,
                    "', '".join(sectionTypes.keys()) # Format a list of the available section types.
                ))
            # Call the appropriate ini-section initialiser for this type.
            sectionTypes[sectionType](sectionConfig)

    def iniCellType(self, section):
        name = section['name']
        self.addCellType(CellType(name))

    def iniLayer(self, section):
        name = section['name']
        self.addLayer(Layer(name))
