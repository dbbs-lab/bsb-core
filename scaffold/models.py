import numpy as np, random
from .morphologies import Morphology as BaseMorphology
from .helpers import ConfigurableClass, dimensions, origin, SortableByAfter
from .exceptions import MissingMorphologyException, AttributeMissingException


class CellType(SortableByAfter):
    def __init__(self, name, placement=None):
        self.name = name
        self.placement = placement

    def validate(self):
        """
            Check whether this CellType is valid to be used in the simulation.
        """
        pass

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.id = scaffoldInstance.configuration.cell_type_map.index(self.name)
        self.validate()

    def set_morphology(self, morphology):
        """
            Set the Morphology class for this cell type.

            :param morphology: Defines the geometrical constraints for the axon and dendrites of the cell type.
            :type morphology: Instance of a subclass of scaffold.morphologies.Morphology
        """
        if not issubclass(type(morphology), BaseMorphology):
            raise Exception(
                "Only subclasses of scaffold.morphologies.Morphology can be used as cell morphologies."
            )
        self.morphology = morphology

    def set_placement(self, placement):
        """
            Set the placement strategy for this cell type.
        """
        self.placement = placement

    @classmethod
    def get_ordered(cls, objects):
        return sorted(objects.values(), key=lambda x: x.placement.get_placement_count())

    def has_after(self):
        return hasattr(self.placement, "after")

    def get_after(self):
        return None if not self.has_after() else self.placement.after

    def create_after(self):
        self.placement.after = []

    def get_placed_count(self):
        return self.scaffold.statistics.cells_placed[self.name]

    def get_ids(self):
        if self.entity:
            dataset = self.scaffold.get_entities_by_type(self.name)
        else:
            dataset = self.scaffold.cells_by_type[self.name][:, 0]
        return np.array(dataset, dtype=int)

    def get_cells(self):
        if self.entity:
            return self.scaffold.get_entities_by_type(self.name)
        return self.scaffold.get_cells_by_type(self.name)

    def list_all_morphologies(self):
        """
            Return a list of all the morphology identifiers that can represent
            this cell type in the simulation volume.
        """
        if not hasattr(self.morphology, "detailed_morphologies"):
            return []
        morphology_config = self.morphology.detailed_morphologies
        # TODO: More selection mechanisms like tags
        if "names" in morphology_config:
            m_names = morphology_config["names"]
            return m_names.copy()
        else:
            raise NotImplementedError(
                "Detailed morphologies can currently only be selected by name."
            )


class Layer(dimensions, origin):
    def __init__(self, name, origin, dimensions, scaling=True):
        # Name of the layer
        self.name = name
        # The XYZ coordinates of the point at the center of the bottom plane of the layer.
        self.origin = np.array(origin)
        # Dimensions in the XYZ axes.
        self.dimensions = np.array(dimensions)
        self.volumeOccupied = 0.0
        # Should this layer scale when the simulation volume is resized?
        self.scaling = scaling

    @property
    def available_volume(self):
        return self.volume - self.volumeOccupied

    @property
    def thickness(self):
        return self.height

    def allocateVolume(volume):
        self.volumeOccupied += volume

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance

    def scale_to_reference(self):
        """
            Compute scaled layer volume

            To compute layer thickness, we scale the current layer to the combined volume
            of the reference layers. A ratio between the dimension can be specified to
            alter the shape of the layer. By default equal ratios are used and a cubic
            layer is obtained (given by `dimension_ratios`).

            The volume of the current layer (= X*Y*Z) is scaled with respect to the volume
            of reference layers by a factor `volume_scale`, so:

            X*Y*Z = volume_reference_layers / volume_scale                [A]

            Supposing that the current layer dimensions (X,Y,Z) are each one depending on
            the dimension Y according to `dimension_ratios`, we obtain:

            X*Y*Z = (Y*dimension_ratios[0] * Y * (Y*dimension_ratios[2])  [B]
            X*Y*Z = (Y^3) * prod(dimension_ratios)                        [C]

            Therefore putting together [A] and [C]:
            (Y^3) * prod(dimension_ratios) = volume_reference_layers / volume_scale

            from which we derive the normalized_size Y, according to the following
            formula:

            Y = cubic_root(volume_reference_layers / volume_scale * prod(dimension_ratios))
        """
        volume_reference_layers = np.sum(
            list(map(lambda layer: layer.volume, self.reference_layers))
        )
        # Compute volume: see docstring.
        normalized_size = pow(
            volume_reference_layers
            / (self.volume_scale * np.prod(self.dimension_ratios)),
            1 / 3,
        )
        # Apply the normalized size with their ratios to each dimension
        self.dimensions = np.multiply(
            np.repeat(normalized_size, 3), self.dimension_ratios
        )


class Resource:
    def __init__(self, handler, path):
        self._handler = handler
        self._path = path

    def get_dataset(self, selector=()):
        with self._handler.load("r") as f:
            return f[self._path][selector]

    @property
    def attributes(self):
        with self._handler.load("r") as f:
            return dict(f[self._path].attrs)

    def get_attribute(self, name):
        attrs = self.attributes
        if name not in attrs:
            raise AttributeMissingException(
                "Attribute '{}' not found in '{}'".format(name, self._path)
            )
        return attrs[name]

    def exists(self):
        with self._handler.load("r") as f:
            return self._path in f

    def unmap(self, selector=(), mapping=lambda m, x: m[x], data=None):
        if data is None:
            data = self.get_dataset(selector)
        map = self.get_attribute("map")
        unmapped = []
        for record in data:
            unmapped.append(mapping(map, record))
        return np.array(unmapped)

    def unmap_one(self, data, mapping=None):
        if mapping is None:
            return self.unmap(data=[data])
        else:
            return self.unmap(data=[data], mapping=mapping)


class Connection:
    def __init__(
        self,
        from_id,
        to_id,
        from_compartment=None,
        to_compartment=None,
        from_morphology=None,
        to_morphology=None,
    ):
        self.from_id = from_id
        self.to_id = to_id
        if (
            from_compartment is not None
            or to_compartment is not None
            or from_morphology is not None
            or to_morphology is not None
        ):
            # If one of the 4 arguments for a detailed connection is given, all 4 are required.
            if (
                from_compartment is None
                or to_compartment is None
                or from_morphology is None
                or to_morphology is None
            ):
                raise Exception(
                    "Insufficient arguments given to Connection constructor."
                    + " If one of the 4 arguments for a detailed connection is given, all 4 are required."
                )
            self.from_compartment = from_morphology.compartments[from_compartment]
            self.to_compartment = to_morphology.compartments[to_compartment]


class ConnectivitySet(Resource):
    def __init__(self, handler, tag):
        super().__init__(handler, "/cells/connections/" + tag)
        self.tag = tag
        self.compartment_set = Resource(handler, "/cells/connection_compartments/" + tag)
        self.morphology_set = Resource(handler, "/cells/connection_morphologies/" + tag)

    @property
    def connections(self):
        return [Connection(c[0], c[1]) for c in self.get_dataset()]

    @property
    def from_identifiers(self):
        return self.get_dataset()[:, 0]

    @property
    def to_identifiers(self):
        return self.get_dataset()[:, 1]

    @property
    def intersections(self):
        if not self.compartment_set.exists():
            raise MissingMorphologyException(
                "No intersection/morphology information for the '{}' connectivity set.".format(
                    self.tag
                )
            )
        else:
            return self.get_intersections()

    def get_intersections(self):
        intersections = []
        morphos = {}

        def _cache_morpho(id):
            # Keep a cache of the morphologies so that all morphologies with the same
            # id refer to the same object, and so that they aren't redundandly loaded.
            id = int(id)
            if not id in morphos:
                name = self.morphology_set.unmap_one(id)[0]
                if isinstance(name, bytes):
                    name = name.decode("UTF-8")
                morphos[id] = self._handler.scaffold.morphology_repository.get_morphology(
                    name
                )

        cells = self.get_dataset()
        for cell_ids, comp_ids, morpho_ids in zip(
            cells, self.compartment_set.get_dataset(), self.morphology_set.get_dataset()
        ):
            from_morpho_id = int(morpho_ids[0])
            to_morpho_id = int(morpho_ids[1])
            # Load morphologies from the map if they're not in the cache yet
            _cache_morpho(from_morpho_id)
            _cache_morpho(to_morpho_id)
            # Append the intersection with a new connection
            intersections.append(
                Connection(
                    *cell_ids,  # zipped dataset: from id & to id
                    *comp_ids,  # zipped morphologyset: from comp & to comp
                    morphos[from_morpho_id],  # cached: from TrueMorphology
                    morphos[to_morpho_id]  # cached: to TrueMorphology
                )
            )
        return intersections

    def __iter__(self):
        if self.compartment_set.exists():
            return self.intersections
        else:
            return self.connections

    @property
    def meta(self):
        """
            Retrieve the metadata associated with this connectivity set. Returns
            ``None`` if the connectivity set does not exist.

            :return: Metadata
            :rtype: dict
        """
        return self.attributes

    @property
    def connection_types(self):
        """
            Return all the ConnectionStrategies that contributed to the creation of this
            connectivity set.
        """
        # Get list of contributing types
        type_list = self.attributes["connection_types"]
        # Map contributing type names to contributing types
        return list(
            map(lambda name: self._handler.scaffold.get_connection_type(name), type_list)
        )


class Cell:
    def __init__(self, id, cell_type, position):
        self.id = int(id)
        self.type = cell_type
        self.position = position

    @classmethod
    def from_repo_data(cls, cell_type, data):
        return cls(data[0], cell_type, data[2:5])


class MorphologySet:
    def __init__(self, scaffold, cell_type, cells, compartment_types=None, N=50):
        self.scaffold = scaffold
        self.cell_type = cell_type
        self._morphology_map = cell_type.list_all_morphologies()
        # Select a random morphology for each cell.
        self._morphology_index = [
            random.choice(range(len(self._morphology_map))) for _ in range(len(cells))
        ]
        # Voxelize a morphology `i` of the available morphologies
        def morpho(i):
            m = scaffold.morphology_repository.get_morphology(self._morphology_map[i])
            m._set_index = i
            m.voxelize(
                N, compartments=m.get_compartments(compartment_types=compartment_types)
            )
            return m

        # Voxelize each morphology
        self._morphologies = [morpho(i) for i in range(len(self._morphology_map))]
        self._cells = [Cell.from_repo_data(cell_type, cell) for cell in cells]

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return zip(self._cells, self._unmap_morphologies())

    def __getitem__(self, id):
        return self._cells[id], self._unmap_morphology(id)

    def _unmap_morphology(self, id):
        return self._morphologies[self._morphology_index[id]]

    def _unmap_morphologies(self):
        return [self._morphologies[i] for i in self._morphology_index]
