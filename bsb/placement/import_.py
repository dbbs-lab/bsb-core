import abc
import csv
import io
import typing
from collections import defaultdict

import numpy as np
import psutil
from tqdm import tqdm

from .. import config
from ..config import refs
from ..exceptions import ConfigurationError
from ..mixins import NotParallel
from ..storage._chunks import Chunk
from .strategy import PlacementStrategy

if typing.TYPE_CHECKING:
    from ..cell_types import CellType
    from ..storage._files import FileDependencyNode
    from ..topology import Partition


@config.node
class ImportPlacement(NotParallel, PlacementStrategy, abc.ABC, classmap_entry=None):
    source: "FileDependencyNode" = config.file(required=True)
    cell_types: list["CellType"] = config.reflist(refs.cell_type_ref, required=False)
    partitions: list["Partition"] = config.reflist(refs.partition_ref, required=False)

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
    x_header: str = config.attr(default="x")
    y_header: str = config.attr(default="y")
    z_header: str = config.attr(default="z")
    type_header: str = config.attr()
    delimiter: str = config.attr(default=",")
    progress_bar: bool = config.attr(type=bool, default=True)

    def __boot__(self):
        if not self.type_header and len(self.get_considered_cell_types()) > 1:
            raise ConfigurationError(
                "Must set `type_header` to import multiple cell types from single CSV."
            )

    def parse_source(self, indicators):
        self._reset_cache()
        chunk_size = np.array(self.scaffold.network.chunk_size)
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
            cts = self.get_considered_cell_types()
            if len(cts) == 1:
                name = cts[0].name
            i = 0
            if self.progress_bar:
                reader = tqdm(reader, desc="imported", unit=" lines")
            for line in reader:
                ct_cache = self._cache[line[type_col] if type_col is not None else name]
                coords = [_safe_float(line[c]) for c in coord_cols]
                others = [_safe_float(line[c]) for c in other_cols]
                cache = ct_cache[tuple(coords // chunk_size)]
                cache[0].append(coords)
                cache[1].append(others)

                if i % 10000 == 0:
                    est_memsize = (len(others) + 3) * i * 8
                    av_mem = psutil.virtual_memory().available
                    if est_memsize > av_mem / 10:
                        print(
                            "FLUSHING, AVAILABLE MEM:",
                        )
                        self._flush(indicators)
                i += 1
            self._flush(indicators)

    def get_considered_cell_types(self):
        return self.cell_types or self.scaffold.cell_types.values()

    def _reset_cache(self):
        self._cache = {
            ct.name: defaultdict(lambda: [[], []])
            for ct in self.get_considered_cell_types()
        }

    def _flush(self, indicators):
        cell_types = self.get_considered_cell_types()
        iter = zip(cell_types, self._cache.values())
        if self.progress_bar:
            iter = tqdm(
                iter,
                desc="cell types",
                total=len(cell_types),
            )
        for ct, chunked_cache in iter:
            inner = ((Chunk(c, None), data) for c, data in chunked_cache.items())
            if self.progress_bar:
                inner = tqdm(
                    inner,
                    desc="saved",
                    total=len(chunked_cache),
                    bar_format="{l_bar}{bar} [ {n_fmt}/{total_fmt} time left: {remaining}, time spent: {elapsed}]",
                )
            for chunk, data in inner:
                additional = np.array(data[1], dtype=float).T
                self.place_cells(
                    indicators[ct.name],
                    np.array(data[0]),
                    chunk,
                    additional={
                        name: col for name, col in zip(self._other_colnames, additional)
                    },
                )
        self._reset_cache()


def _safe_float(value: typing.Any) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


__all__ = ["CsvImportPlacement", "ImportPlacement"]
