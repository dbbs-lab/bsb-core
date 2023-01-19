import abc
import csv
import io
from collections import defaultdict

import psutil
import numpy as np

from .strategy import PlacementStrategy
from ..storage import Chunk
from ..exceptions import ConfigurationError
from .. import config
from ..config import refs
from ..mixins import NotParallel


@config.node
class ImportPlacement(NotParallel, PlacementStrategy, abc.ABC, classmap_entry=None):
    source = config.file(required=True)
    cell_types = config.reflist(refs.cell_type_ref, required=False)
    partitions = config.reflist(refs.partition_ref, required=False)

    @config.property(default=False)
    def cache(self):
        return self.source.cache

    @cache.setter
    def cache(self, value):
        self.source.cache = bool(value)

    def place(self, chunk, indicators):
        self.parse_source(indicators)

    @abc.abstractmethod
    def parse_source(self, indicators):
        pass


@config.node
class CsvImportPlacement(ImportPlacement):
    x_header = config.attr(default="x")
    y_header = config.attr(default="y")
    z_header = config.attr(default="z")
    type_header = config.attr()
    delimiter = config.attr(default=",")

    def __boot__(self):
        if not self.type_header and len(self.get_considered_cell_types()) > 1:
            raise ConfigurationError(
                "Must set `type_header` to import multiple cell types from single CSV."
            )

    def parse_source(self, indicators):
        self._reset_cache()
        chunk_size = self.scaffold.network.chunk_size
        print(psutil.virtual_memory())
        with self.source.provide_stream() as (fp, encoding):
            text = io.TextIOWrapper(fp, encoding=encoding, newline="")
            reader = csv.reader(text)
            headers = [h.strip().lower() for h in next(reader)]
            type_col = headers.index(self.type_header) if self.type_header else None
            coord_cols = [
                *map(headers.index, (self.x_header, self.y_header, self.z_header))
            ]
            other_cols = [
                r for r in range(len(headers)) if r not in coord_cols and r != type_col
            ]
            self._other_colnames = [headers[c] for c in other_cols]
            i = 0
            for line in reader:
                ct_cache = self._cache[line[type_col] if type_col is not None else 0]
                coords = [line[c] for c in coord_cols]
                others = [line[c] for c in other_cols]
                cache = ct_cache[tuple(coords // chunk_size)]
                cache[0].append(coords)
                cache[1].append(others)

                if i % 100000 == 0:
                    est_memsize = (len(cache[1][0]) + 3) * len(cache[0]) * 8
                    av_mem = psutil.virtual_memory().available
                    if est_memsize > av_mem / 10:
                        print(
                            "FLUSHING, AVAILABLE MEM:",
                        )
                        self._flush()
                i += 1
            self._flush()

    def get_considered_cell_types(self):
        return self.cell_types or self.scaffold.cell_types.values()

    def _reset_cache(self):
        self._cache = {
            ct.name: defaultdict(lambda: [[], []])
            for ct in self.get_considered_cell_types()
        }

    def _flush(self, indicators):
        for ct, chunked_cache in zip(
            self.get_considered_cell_types(), self._cache.values()
        ):
            for chunk, data in (
                (Chunk(c, None), data) for c, data in chunked_cache.items()
            ):
                additional = np.array(data[1], dtype=float)
                self.place_cells(
                    indicators[ct],
                    np.array(data[0]),
                    chunk,
                    additional={
                        name: col for name, col in zip(self._other_colnames, data[1])
                    },
                )
        self._reset_cache()

    # @abc.abstractmethod
    # def _place_from_csv(self):
    #     src = self.get_external_source()
    #     if not self.check_external_source():
    #         raise MissingSourceError(f"Missing source file '{src}' for `{self.name}`.")
    #     # If the `map_header` is given, we should store all data in that column
    #     # as references that the user will need later on to map their external
    #     # data to our generated data
    #     should_map = self.map_header is not None
    #     # Read the CSV file's first line to get the column names
    #     with open(src, "r") as f:
    #         headers = f.readline().split(self.delimiter)
    #     # Search for the x, y, z headers
    #     usecols = list(map(headers.index, (self.x_header, self.y_header, self.z_header)))
    #     if should_map:
    #         # Optionally, search for the map header
    #         usecols.append(headers.index(self.map_header))
    #     # Read the entire csv, skipping the headers and only the cols we need.
    #     data = np.loadtxt(
    #         src,
    #         usecols=usecols,
    #         skiprows=1,
    #         delimiter=self.delimiter,
    #     )
    #     if should_map:
    #         # If a map column was appended, slice it off
    #         external_map = data[:, -1]
    #         data = data[:, :-1]
    #         # Check for garbage
    #         duplicates = len(external_map) - len(np.unique(external_map))
    #         if duplicates:
    #             raise SourceQualityError(f"{duplicates} duplicates in source '{src}'")
    #         # And store it as appendix dataset
    #         self.scaffold.append_dset(self.name + "_ext_map", external_map)
    #     # Store the CSV positions in the scaffold
    #     self.scaffold.place_cells(self.cell_type, None, data)
