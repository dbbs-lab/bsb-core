import numpy as np
from .morphologies import Morphology as BaseMorphology
from .helpers import ConfigurableClass, dimensions, origin, SortableByAfter
from .exceptions import MissingMorphologyException, AttributeMissingException

class CellType(SortableByAfter):

    def __init__(self, name, placement=None):
        self.name = name
        self.placement = placement

    def validate(self):
        '''
            Check whether this CellType is valid to be used in the simulation.
        '''
        pass

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.id = scaffoldInstance.configuration.cell_type_map.index(self.name)
        self.validate()

    def set_morphology(self, morphology):
        '''
            Set the Morphology class for this cell type.

            :param morphology: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type morphology: Instance of a subclass of scaffold.morphologies.Morphology
        '''
        if not issubclass(type(morphology), BaseMorphology):
            raise Exception("Only subclasses of scaffold.morphologies.Morphology can be used as cell morphologies.")
        self.morphology = morphology

    def set_placement(self, placement):
        '''
            Set the placement strategy for this cell type.
        '''
        self.placement = placement

    @classmethod
    def get_ordered(cls, objects):
        return sorted(objects.values(), key=lambda x: x.placement.get_placement_count(x))

    def has_after(self):
        return hasattr(self.placement, "after")

    def get_after(self):
        return None if not self.has_after() else self.placement.after

    def get_placed_count(self):
        return self.scaffold.statistics.cells_placed[self.name]

    def get_ids(self):
        return np.array(self.scaffold.cells_by_type[self.name][:,0], dtype=int)

    def list_all_morphologies(self):
        if not hasattr(self.morphology, "detailed_morphologies"):
            return []
        morphology_config = self.morphology.detailed_morphologies
        # TODO: More selection mechanisms like tags
        if 'names' in morphology_config:
            m_names = morphology_config['names']
            return m_names
        else:
            raise NotImplementedError("Detailed morphologies can currently only be selected by name.")

class Layer(dimensions, origin):

    def __init__(self, name, origin, dimensions, scaling=True):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = np.array(origin)
        # Dimensions in the XYZ axes.
        self.dimensions = np.array(dimensions)
        self.volumeOccupied = 0.
        # Should this layer scale when the simulation volume is resized?
        self.scaling = scaling

    @property
    def available_volume(self):
        return self.volume - self.volumeOccupied

    @property
    def thickness(self):
        return self.dimensions[1]

    def allocateVolume(volume):
        self.volumeOccupied += volume

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance

class Resource:
    def __init__(self, handler, path):
        self.handler = handler
        self.path = path

    def get_dataset(self, selector=()):
        with self.handler.load("r") as f:
            return f[self.path][selector]

    @property
    def attributes(self):
        with self.handler.load("r") as f:
            return dict(f[self.path].attrs)

    def get_attribute(self, name):
        attrs = self.attributes
        if not name in attrs:
            raise AttributeMissingException("Attribute '{}' not found in '{}'".format(name, self.path))
        return attrs[name]

    def exists(self):
        with self.handler.load("r") as f:
            return self.path in f

    def unmap(self, selector=(), mapping=lambda m, x: m[x], data=None):
        if data is None:
            data = self.get_dataset(selector)
        map = self.get_attribute('map')
        unmapped = []
        for record in data:
            print(record)
            unmapped.append(mapping(map, record))
        return np.array(unmapped)

    def unmap_one(self, data, mapping=None):
        if mapping is None:
            return self.unmap(data=[data])
        else:
            return self.unmap(data=[data], mapping=mapping)

class Connection:
    def __init__(self, from_id, to_id, from_compartment, to_compartment, from_morphology, to_morphology):
        self.from_id = from_id
        self.to_id = to_id
        self.from_compartment = from_morphology.compartments[from_compartment]
        self.to_compartment = from_morphology.compartments[to_compartment]

class ConnectivitySet(Resource):
    def __init__(self, handler, tag):
        super().__init__(handler, '/cells/connections/' + tag)
        self.compartment_set = Resource(handler, '/cells/connection_compartments/' + tag)
        self.morphology_set = Resource(handler, '/cells/connection_morphologies/' + tag)

    @property
    def connections(self):
        return self.get_dataset()

    @property
    def intersections(self):
        if not self.compartment_set.exists():
            raise MissingMorphologyException("No intersection/morphology information for this connectivity set.")
        else:
            return self.get_intersections()

    def get_intersections(self):
        intersections = []
        morphos = {}
        cells = self.get_dataset()
        for cell_ids, comp_ids, morpho_ids in zip(cells, self.compartment_set.get_dataset(), self.morphology_set.get_dataset()):
            if not int(morpho_ids[0]) in morphos:
                print('loading morpho')
                name = self.morphology_set.unmap_one(int(morpho_ids[0]))[0].decode('UTF-8')
                print('loaded', name)
                morphos[int(morpho_ids[0])] = self.handler.scaffold.morphology_repository.get_morphology(name)
            if not int(morpho_ids[1]) in morphos:
                print('loading morpho')
                name = self.morphology_set.unmap_one(int(morpho_ids[1]))[0].decode('UTF-8')
                print('loaded', name)
                morphos[int(morpho_ids[1])] = self.handler.scaffold.morphology_repository.get_morphology(name)
            intersections.append(Connection(*cell_ids, *comp_ids, morphos[int(morpho_ids[0])], morphos[int(morpho_ids[1])]))
        return intersections
