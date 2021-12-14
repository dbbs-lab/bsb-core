import numpy as np, random
from .morphologies import Morphology as BaseMorphology, NilCompartment
from .helpers import (
    ConfigurableClass,
    dimensions,
    origin,
    SortableByAfter,
    continuity_list,
    expand_continuity_list,
    count_continuity_list,
    iterate_continuity_list,
)
from .exceptions import *


class CellType(SortableByAfter):
    """
    A CellType represents a population of cells.
    """

    def __init__(self, name, placement=None):
        self.name = name
        self.placement = placement
        self.relay = False

    def validate(self):
        """
        Check whether this CellType is valid to be used in the simulation.
        """
        pass

    def initialise(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.validate()

    def set_morphology(self, morphology):
        """
        Set the Morphology class for this cell type.

        :param morphology: Defines the geometrical constraints for the axon and dendrites of the cell type.
        :type morphology: Instance of a subclass of scaffold.morphologies.Morphology
        """
        if not issubclass(type(morphology), BaseMorphology):
            raise ClassError(
                "Only subclasses of scaffold.morphologies.Morphology can be used as cell morphologies."
            )
        self.morphology = morphology

    def set_placement(self, placement):
        """
        Set the placement strategy for this cell type.
        """
        self.placement = placement

    def place(self):
        """
        Place this cell type.
        """
        self.scaffold.place_cell_type(self)

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

    def _get_cached_ids(self):
        if self.entity:
            dataset = self.scaffold.entities_by_type[self.name]
        else:
            dataset = self.scaffold.cells_by_type[self.name][:, 0]
        return np.array(dataset, dtype=int)

    def _ser_cached_ids(self):
        raw_ids = self._get_cached_ids()
        return continuity_list(raw_ids)

    def get_cells(self):
        if self.entity:
            return self.scaffold.get_entities_by_type(self.name)
        return self.scaffold.get_cells_by_type(self.name)

    def list_all_morphologies(self):
        """
        Return a list of all the morphology identifiers that can represent
        this cell type in the simulation volume.
        """
        if not hasattr(self, "morphology") or not hasattr(
            self.morphology, "detailed_morphologies"
        ):
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

    def get_placement_set(self):
        return self.scaffold.get_placement_set(self)


class Layer(dimensions, origin):
    """
    A Layer represents a compartment of the topology of the simulation volume that slices
    the volume in horizontally stacked portions.
    """

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

        Y = cubic_root((volume_reference_layers * volume_scale) / prod(dimension_ratios))
        """
        volume_reference_layers = np.sum(
            list(map(lambda layer: layer.volume, self.reference_layers))
        )
        # Compute volume: see docstring.
        normalized_size = pow(
            volume_reference_layers * self.volume_scale / np.prod(self.dimension_ratios),
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

    def get_dataset(self, selector=(), dtype=None):
        with self._handler.load("r") as f:
            if not self._path in f():
                raise DatasetNotFoundError(
                    "Dataset '{}' not found in '{}'.".format(
                        self._path, self._handler.file
                    )
                )
            d = f()[self._path][selector]
            if dtype:
                d = d.astype(dtype)
            return d

    @property
    def attributes(self):
        with self._handler.load("r") as f:
            return dict(f()[self._path].attrs)

    def get_attribute(self, name):
        attrs = self.attributes
        if name not in attrs:
            raise AttributeMissingError(
                "Attribute '{}' not found in '{}'".format(name, self._path)
            )
        return attrs[name]

    def exists(self):
        with self._handler.load("r") as f:
            return self._path in f()

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

    def __iter__(self):
        return iter(self.get_dataset())

    @property
    def shape(self):
        with self._handler.load("r") as f:
            return f()[self._path].shape


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
            if from_compartment < -1:
                raise RuntimeError("Invalid compartment data")
            elif from_compartment == -1:
                self.from_compartment = NilCompartment()
            else:
                self.from_compartment = from_morphology.compartments[from_compartment]

            if to_compartment < -1:
                raise RuntimeError("Invalid compartment data")
            elif to_compartment == -1:
                self.to_compartment = NilCompartment()
            else:
                self.to_compartment = to_morphology.compartments[to_compartment]


class ConnectivitySet(Resource):
    """
    Connectivity sets store connections.
    """

    def __init__(self, handler, tag):
        super().__init__(handler, "/cells/connections/" + tag)
        if not self.exists():
            raise DatasetNotFoundError("ConnectivitySet '{}' does not exist".format(tag))
        self.scaffold = handler.scaffold
        self.tag = tag
        self.compartment_set = Resource(handler, "/cells/connection_compartments/" + tag)
        self.morphology_set = Resource(handler, "/cells/connection_morphologies/" + tag)

    def has_compartment_data(self):
        """
        Check if compartment data exists for this connectivity set.
        """
        return self.compartment_set.exists()

    def is_orphan(self):
        return not bool(self.attributes["connection_types"])

    @property
    def connections(self):
        """
        Return a list of :class:`Intersections <.models.Connection>`. Connections
        contain pre- & postsynaptic identifiers.
        """
        return [Connection(c[0], c[1]) for c in self.get_dataset()]

    @property
    def from_identifiers(self):
        """
        Return a list with the presynaptic identifier of each connection.
        """
        return self.get_dataset(dtype=int)[:, 0]

    @property
    def to_identifiers(self):
        """
        Return a list with the postsynaptic identifier of each connection.
        """
        return self.get_dataset(dtype=int)[:, 1]

    @property
    def intersections(self):
        """
        Return a list of :class:`Intersections <.models.Connection>`. Intersections
        contain pre- & postsynaptic identifiers and the intersecting compartments.
        """
        return self.get_intersections()

    def get_intersections(self):
        intersections = []
        morphos = {-1: None}

        def _cache_morpho(id):
            # Keep a cache of the morphologies so that all morphologies with the same
            # id refer to the same object, and so that they aren't redundandly loaded.
            id = int(id)
            if not id in morphos:
                name = self.morphology_set.unmap_one(id)[0]
                if isinstance(name, bytes):
                    name = name.decode("UTF-8")
                morphos[id] = self.scaffold.morphology_repository.get_morphology(name)

        cells = self.get_dataset()
        if self.has_compartment_data():
            comp_data = self.compartment_set.get_dataset()
            morpho_data = self.morphology_set.get_dataset()
        else:
            comp_data = np.ones(cells.shape) * -1
            morpho_data = np.ones(cells.shape) * -1

        for cell_ids, comp_ids, morpho_ids in zip(cells, comp_data, morpho_data):
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
                    morphos[from_morpho_id],  # cached: 'from' TrueMorphology
                    morphos[to_morpho_id]  # cached: 'to' TrueMorphology
                )
            )
        return intersections

    def get_divergence_list(self):
        presynaptic_type = self.get_presynaptic_types()[0]
        placement_set = self.scaffold.get_placement_set(presynaptic_type)
        unique_connections = np.unique(self.get_dataset(), axis=0)
        _, divergence_list = np.unique(unique_connections[:, 0], return_counts=True)
        return np.concatenate(
            (divergence_list, np.zeros(len(placement_set) - len(divergence_list)))
        )

    @property
    def divergence(self):
        divergence_list = self.get_divergence_list()
        if len(divergence_list) == 0:
            return 0
        return np.mean(divergence_list)

    def get_convergence_list(self):
        postsynaptic_type = self.get_postsynaptic_types()[0]
        placement_set = self.scaffold.get_placement_set(postsynaptic_type)
        unique_connections = np.unique(self.get_dataset(), axis=0)
        _, convergence_list = np.unique(unique_connections[:, 1], return_counts=True)
        return np.concatenate(
            (convergence_list, np.zeros(len(placement_set) - len(convergence_list)))
        )

    @property
    def convergence(self):
        convergence_list = self.get_convergence_list()
        if len(convergence_list) == 0:
            return 0
        return np.mean(convergence_list)

    def __iter__(self):
        if self.compartment_set.exists():
            return self.intersections
        else:
            return self.connections

    def __len__(self):
        return self.shape[0]

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
        return list(map(lambda name: self.scaffold.get_connection_type(name), type_list))

    def _get_cell_types(self, key="from"):
        meta = self.meta
        if key + "_cell_types" in meta:
            cell_types = set()
            for name in meta[key + "_cell_types"]:
                cell_types.add(self.scaffold.get_cell_type(name))
            return list(cell_types)
        cell_types = set()
        for connection_type in self.connection_types:
            cell_types |= set(connection_type.__dict__[key + "_cell_types"])
        return list(cell_types)

    def get_presynaptic_types(self):
        """
        Return a list of the presynaptic cell types found in this set.
        """
        return self._get_cell_types(key="from")

    def get_postsynaptic_types(self):
        """
        Return a list of the postsynaptic cell types found in this set.
        """
        return self._get_cell_types(key="to")


class PlacementSet(Resource):
    """
    Fetches placement data from storage. You can either access the parallel-array
    datasets ``.identifiers``, ``.positions`` and ``.rotations`` individually or
    create a collection of :class:`Cells <.models.Cell>` that each contain their own
    identifier, position and rotation.

    .. note::

        Use :func:`.core.get_placement_set` to correctly obtain a PlacementSet.
    """

    def __init__(self, handler, cell_type):
        root = "/cells/placement/"
        tag = cell_type.name
        super().__init__(handler, root + tag)
        if not self.exists():
            raise DatasetNotFoundError("PlacementSet '{}' does not exist".format(tag))
        self.type = cell_type
        self.tag = tag
        self._identifiers = Resource(handler, root + tag + "/identifiers")
        self._filter = f = _Filter()

        def id_source():
            return np.array(
                expand_continuity_list(self._identifiers.get_dataset()), dtype=int
            )

        self._filter.filter_source = id_source
        self.identifier_set = _FilteredIds(handler, root + tag + "/identifiers", f)
        self.positions_set = _FilteredResource(handler, root + tag + "/positions", f)
        self.rotation_set = _FilteredResource(handler, root + tag + "/rotations", f)

    @property
    def identifiers(self):
        """
        Return a list of cell identifiers.
        """
        return self.identifier_set.get_dataset()

    @property
    def positions(self):
        """
        Return a dataset of cell positions.
        """
        try:
            return self.positions_set.get_dataset()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No position information for the '{}' placement set.".format(self.tag)
            )

    @property
    def rotations(self):
        """
        Return a dataset of cell rotations.

        :raises: DatasetNotFoundError when there is no rotation information for this
           cell type.
        """
        try:
            return self.rotation_set.get_dataset()
        except DatasetNotFoundError:
            raise DatasetNotFoundError(
                "No rotation information for the '{}' placement set.".format(self.tag)
            )

    @property
    def cells(self):
        """
        Reorganize the available datasets into a collection of :class:`Cells
        <.models.Cell>`
        """
        return [
            Cell(id, self.type, position, rotation) for id, position, rotation in self
        ]

    def __iter__(self):
        id_iter = iterate_continuity_list(self._identifiers.get_dataset())
        iterators = [iter(id_iter), self._none(), self._none()]
        if self.positions_set.exists():
            iterators[1] = iter(self.positions)
        if self.rotation_set.exists():
            iterators[2] = iter(self.rotations)
        return zip(*iterators)

    def __len__(self):
        return count_continuity_list(self._identifiers)

    def _none(self):
        """
        Generate ``len(self)`` times ``None``
        """
        for i in range(len(self)):
            yield None

    def set_filter(self, filter):
        self._filter.active_filter = filter


class _Filter:
    """
    To use, set an `active_filter` and `filter_source` function that return numpy arrays.

    `filter_source` should return a dataset of the same shape as the `data` being filtered.
    `active_filter` will then be called to create a boolean mask from `filter_source`,
    applied as a filter on the data being filtered. (This means that `filter_source` and
    `data` should be parallel arrays)
    """

    active_filter = None
    filter_source = None

    def filter(self, data):
        if self.active_filter is None:
            return data
        return data[np.isin(self.filter_source(), self.active_filter())]


class _FilteredResource(Resource):
    def __init__(self, handler, path, filter):
        super().__init__(handler, path)
        self._filter = filter

    def get_dataset(self, *args, **kwargs):
        return self._filter.filter(super().get_dataset(*args, **kwargs))


class _FilteredIds(_FilteredResource):
    def get_dataset(self, *args, **kwargs):
        data = np.array(
            expand_continuity_list(Resource.get_dataset(self, *args, **kwargs)), dtype=int
        )
        return self._filter.filter(data)


class Cell:
    def __init__(self, id, cell_type, position, rotation=None):
        self.id = int(id)
        self.type = cell_type
        self.position = position
        self.rotation = rotation

    @classmethod
    def from_repo_data(cls, cell_type, data):
        return cls(data[0], cell_type, data[2:5])


class MorphologySet:
    def __init__(self, scaffold, cell_type, placement_set, compartment_types=None, N=50):
        self.scaffold = scaffold
        self.cell_type = cell_type

        self._construct_map(cell_type, placement_set, compartment_types, N)

        self._placement_set = placement_set
        self._cells = placement_set.cells

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

    def _construct_map(self, cell_type, placement_set, compartment_types=None, N=50):
        """
        Associate to the placement_set an index map to only the morphologies
        in the MorphologyRepository needed for that placement set

        """
        # Fetch a list of all available morphology names, whose index in the
        # list will be used as an identifier for the morphology in the cache
        # list of all morphologies
        morphology_names = self.scaffold.morphology_repository.list_morphologies(
            cell_type=cell_type
        )

        if len(morphology_names) == 0:
            raise MorphologyRepositoryError("No morphologies found for " + cell_type.name)

        # Select a random morphology for each cell and store its index in a list
        random_morphologies = [
            random.choice(range(len(morphology_names))) for _ in range(len(placement_set))
        ]

        self._morphology_index = []
        self._morphology_map = []
        if placement_set.rotation_set.exists() or (
            self.scaffold and hasattr(self.scaffold.rotations, cell_type.name)
        ):
            # Rotations? Get the rotated version of the randomly selected morphology and
            # check in `self._morphology_map` if it has been used before.
            rotations = (
                placement_set.rotation_set.get_dataset()
                if placement_set.rotation_set.exists()
                else self.scaffold.rotations[cell_type.name]
            )
            # Get the names of the rotated morphologies that need to be loaded
            for i in range(len(rotations)):
                rot_str = (
                    "__" + str(int(rotations[i][0])) + "_" + str(int(rotations[i][1]))
                )
                mname = morphology_names[random_morphologies[i]] + rot_str

                if mname in self._morphology_map:
                    self._morphology_index.append(self._morphology_map.index(mname))
                else:
                    self._morphology_index.append(len(self._morphology_map))
                    self._morphology_map.append(mname)
        else:
            # No rotations? Just use the randomly selected morphologies
            self._morphology_index = random_morphologies
            self._morphology_map = morphology_names

        # Function to load and voxelize a morphology
        def load_morpho(scaffold, morpho_ind, compartment_types=None):
            m = scaffold.morphology_repository.get_morphology(
                self._morphology_map[morpho_ind]
            )
            m._set_index = morpho_ind
            m.voxelize(N, compartments=m.get_compartments(compartment_types))
            return m

        # Load and voxelize only the unique morphologies present in the morphology map.
        self._morphologies = [
            load_morpho(self.scaffold, i, compartment_types)
            for i in range(len(self._morphology_map))
        ]
