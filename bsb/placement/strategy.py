from .. import config
from ..exceptions import *
from ..reporting import report, warn
from ..config import refs, types
from ..helpers import SortableByAfter
from ..morphologies import MorphologySet
from ..storage import Chunk
from .indicator import PlacementIndications, PlacementIndicator
from .distributor import DistributorsNode
import numpy as np
import itertools
import abc
import os


@config.dynamic(attr_name="strategy", required=True)
class PlacementStrategy(abc.ABC, SortableByAfter):
    """
    Quintessential interface of the placement module. Each placement strategy defines an
    approach to placing neurons into a volume.
    """

    name = config.attr(key=True)
    cell_types = config.reflist(refs.cell_type_ref, required=True)
    partitions = config.reflist(refs.partition_ref, required=True)
    overrides = config.dict(type=PlacementIndications)
    after = config.reflist(refs.placement_ref)
    distribute = config.attr(type=DistributorsNode, default=dict, call_default=True)
    indicator_class = PlacementIndicator

    def __boot__(self):
        self._queued_jobs = []

    @abc.abstractmethod
    def place(self, chunk, indicators):
        """
        Central method of each placement strategy. Given a chunk, should fill that chunk
        with cells by calling the scaffold's (available as ``self.scaffold``)
        :func:`~bsb.core.Scaffold.place_cells` method.
        """
        pass

    def place_cells(self, indicator, positions, chunk):
        distr_ = self.distribute._curry(self.partitions, indicator, positions)

        if indicator.use_morphologies():
            try:
                morphologies = distr_("morphologies")
            except EmptySelectionError as e:
                selectors = ", ".join(f"{s}" for s in e.selectors)
                raise DistributorError(
                    "%property% distribution of `%strategy.name%` couldn't find any"
                    + f" morphologies with the following selector(s): {selectors}",
                    "Morphology",
                    self,
                ) from None
            # Did the morphology distributor give multiple return values?
            if isinstance(morphologies, tuple):
                # Yes, unpack them to morphologies and rotations.
                try:
                    morphologies, rotations = morphologies
                except TypeError:
                    raise ValueError(
                        "Morphology distributors may only return tuples when they are"
                        + " to be unpacked as (morphologies, rotations)"
                    ) from None
                # If a RotationDistributor is not `Implicit`, we override the
                # MorphologyDistributor's rotations.
                if not isinstance(self.distribute.rotations, Implicit):
                    rotations = distr_("rotations")
            else:
                # No, distribute the rotations.
                rotations = distr_("rotations")
        else:
            morphologies, rotations = None, None

        additional = {prop: curry(prop) for prop in self.distribute.properties.keys()}
        self.scaffold.place_cells(
            indicator.cell_type,
            positions=positions,
            rotations=rotations,
            morphologies=morphologies,
            additional=additional,
            chunk=chunk,
        )

    def queue(self, pool, chunk_size):
        """
        Specifies how to queue this placement strategy into a job pool. Can be overridden,
        the default implementation asks each partition to chunk itself and creates 1
        placement job per chunk.
        """
        # Reset jobs that we own
        self._queued_jobs = []
        # Get the queued jobs of all the strategies we depend on.
        deps = set(itertools.chain(*(strat._queued_jobs for strat in self.get_after())))
        for p in self.partitions:
            chunks = p.to_chunks(chunk_size)
            for chunk in chunks:
                job = pool.queue_placement(self, Chunk(chunk, chunk_size), deps=deps)
                self._queued_jobs.append(job)
        report(f"Queued {len(self._queued_jobs)} jobs for {self.name}", level=2)

    def is_entities(self):
        return "entities" in self.__class__.__dict__ and self.__class__.entities

    def get_indicators(self):
        """
        Return indicators per cell type. Indicators collect all configuration information
        into objects that can produce guesses as to how many cells of a type should be
        placed in a volume.
        """
        return {
            ct.name: self.__class__.indicator_class(self, ct) for ct in self.cell_types
        }

    @classmethod
    def get_ordered(cls, objects):
        # No need to sort placement strategies, just obey dependencies.
        return objects

    def guess_cell_count(self):
        return sum(ind.guess() for ind in self.get_indicators().values())

    def has_after(self):
        return hasattr(self, "after")

    def get_after(self):
        return [] if not self.has_after() else self.after

    def create_after(self):
        # I think the reflist should always be there.
        pass


@config.node
class FixedPositions(PlacementStrategy):
    positions = config.attr(type=np.array)

    def place(self, chunk, indicators):
        for indicator in indicators:
            ct = indicator.cell_type
            self.place_cells(ct, indicator, self.positions, chunk)

    def guess_cell_count(self):
        return len(self.positions)


class Entities(PlacementStrategy):
    """
    Implementation of the placement of entities that do not have a 3D position,
    but that need to be connected with other cells of the network.
    """

    entities = True

    def queue(self, pool, chunk_size):
        # Entities ignore chunks since they don't intrinsically store any data.
        pool.queue_placement(self, Chunk([0, 0, 0], chunk_size))

    def place(self, chunk, indicators):
        for indicator in indicators.values():
            cell_type = indicator.cell_type
            # Guess total number, not chunk number, as entities bypass chunking.
            n = indicator.guess()
            self.scaffold.create_entities(cell_type, n)


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
