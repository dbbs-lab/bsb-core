from ..exceptions import *
from ..helpers import ConfigurableClass
import abc
from ..exceptions import *
from ..reporting import report, warn
import numpy as np, os


class PlacementStrategy(ConfigurableClass):
    def __init__(self, cell_type):
        super().__init__()
        self.cell_type = cell_type
        self.layer = None
        self.radius = None
        self.density = None
        self.planar_density = None
        self.placement_count_ratio = None
        self.density_ratio = None
        self.placement_relative_to = None
        self.count = None

    @abc.abstractmethod
    def place(self):
        pass

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    @abc.abstractmethod
    def get_placement_count(self):
        pass


class MightBeRelative:
    """
    Validation class for PlacementStrategies that can be configured relative to other
    cell types.
    """

    def validate(self):
        if self.placement_relative_to is not None:
            # Store the relation.
            self.relation = self.scaffold.configuration.cell_types[
                self.placement_relative_to
            ]
            if self.density_ratio is not None and self.relation.placement.layer is None:
                # A layer volume is required for relative density calculations.
                raise ConfigurationError(
                    "Cannot place cells relative to the density of a placement strategy that isn't tied to a layer."
                )

    def get_relative_count(self):
        # Get the placement count of the ratio cell type and multiply their count by the ratio.
        return int(
            self.relation.placement.get_placement_count() * self.placement_count_ratio
        )

    def get_relative_density_count(self):
        # Get the density of the ratio cell type and multiply it by the ratio.
        ratio = placement.placement_count_ratio
        n1 = self.relation.placement.get_placement_count()
        V1 = self.relation.placement.layer_instance.volume
        V2 = layer.volume
        return int(n1 * ratio * V2 / V1)


class MustBeRelative(MightBeRelative):
    """
    Validation class for PlacementStrategies that must be configured relative to other
    cell types.
    """

    def validate(self):
        if (
            not hasattr(self, "placement_relative_to")
            or self.placement_relative_to is None
        ):
            raise ConfigurationError(
                "The {} requires you to configure another cell type under `placement_relative_to`."
            )
        super().validate()


class Layered(MightBeRelative):
    """
    Class for placement strategies that depend on Layer objects.
    """

    def validate(self):
        super().validate()
        # Check if the layer is given and exists.
        config = self.scaffold.configuration
        if not hasattr(self, "layer"):
            raise AttributeMissingError(
                "Required attribute 'layer' missing from {}".format(self.name)
            )
        if self.layer not in config.layers:
            raise LayerNotFoundError(
                "Unknown layer '{}' in {}".format(self.layer, self.name)
            )
        self.layer_instance = self.scaffold.configuration.layers[self.layer]
        if hasattr(self, "y_restriction"):
            self.restriction_minimum = float(self.y_restriction[0])
            self.restriction_maximum = float(self.y_restriction[1])
        else:
            self.restriction_minimum = 0.0
            self.restriction_maximum = 1.0
        self.restriction_factor = self.restriction_maximum - self.restriction_minimum

    def get_placement_count(self):
        """
        Get the placement count proportional to the available volume in the layer
        times the cell type density.
        """
        layer = self.layer_instance
        available_volume = layer.available_volume
        placement = self.cell_type.placement
        if placement.count is not None:
            return int(placement.count)
        if placement.placement_count_ratio is not None:
            return self.get_relative_count()
        if placement.density_ratio is not None:
            return self.get_relative_density_count()
        if placement.planar_density is not None:
            # Calculate the planar density
            return int(layer.width * layer.depth * placement.planar_density)
        if hasattr(self, "restriction_factor"):
            # Add a restriction factor to the available volume
            return int(available_volume * self.restriction_factor * placement.density)
        # Default: calculate N = V * C
        return int(available_volume * placement.density)


class FixedPositions(Layered, PlacementStrategy):
    casts = {"positions": np.array}

    def place(self):
        self.scaffold.place_cells(self.cell_type, self.layer_instance, self.positions)

    def get_placement_count(self):
        return len(self.positions)


class Entities(Layered, PlacementStrategy):
    """
    Implementation of the placement of entities (e.g., mossy fibers) that do not have
    a 3D position, but that need to be connected with other cells of the scaffold.
    """

    entities = True

    def place(self):
        # Variables
        cell_type = self.cell_type
        scaffold = self.scaffold

        # Get the number of cells that belong in the available volume.
        n_cells_to_place = self.get_placement_count()
        if n_cells_to_place == 0:
            warn(
                "Volume or density too low, no '{}' cells will be placed".format(
                    cell_type.name
                ),
                PlacementWarning,
            )

        scaffold.create_entities(cell_type, n_cells_to_place)


class ExternalPlacement(PlacementStrategy):
    required = ["source"]
    casts = {"format": str, "warn_missing": bool}
    defaults = {
        "format": "csv",
        "x_header": "x",
        "y_header": "y",
        "z_header": "z",
        "map_header": None,
        "warn_missing": True,
        "delimiter": ",",
    }

    has_external_source = True

    def check_external_source(self):
        return os.path.exists(self.source)

    def get_external_source(self):
        return self.source

    def validate(self):
        if self.warn_missing and not self.check_external_source():
            src = self.get_external_source()
            warn(f"Missing external source '{src}' for '{self.name}'")

    def place(self):
        if self.format == "csv":
            return self._place_from_csv()

    def get_placement_count(self):
        import csv

        file = self.get_external_source()
        f = csv.reader(open(file, "r"))
        csv_lines = sum(1 for line in f)
        return csv_lines - 1

    def _place_from_csv(self):
        src = self.get_external_source()
        if not self.check_external_source():
            raise MissingSourceError(f"Missing source file '{src}' for `{self.name}`.")
        # If the `map_header` is given, we should store all data in that column
        # as references that the user will need later on to map their external
        # data to our generated data
        should_map = self.map_header is not None
        # Read the CSV file's first line to get the column names
        with open(src, "r") as f:
            headers = f.readline().split(self.delimiter)
        # Search for the x, y, z headers
        usecols = list(map(headers.index, (self.x_header, self.y_header, self.z_header)))
        if should_map:
            # Optionally, search for the map header
            usecols.append(headers.index(self.map_header))
        # Read the entire csv, skipping the headers and only the cols we need.
        data = np.loadtxt(
            src,
            usecols=usecols,
            skiprows=1,
            delimiter=self.delimiter,
        )
        if should_map:
            # If a map column was appended, slice it off
            external_map = data[:, -1]
            data = data[:, :-1]
            # Check for garbage
            duplicates = len(external_map) - len(np.unique(external_map))
            if duplicates:
                raise SourceQualityError(f"{duplicates} duplicates in source '{src}'")
            # And store it as appendix dataset
            self.scaffold.append_dset(self.name + "_ext_map", external_map)
        # Store the CSV positions in the scaffold
        self.scaffold.place_cells(self.cell_type, None, data)
