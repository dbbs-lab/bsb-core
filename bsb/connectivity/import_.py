import abc
import csv
import io
import typing

import psutil
from tqdm import tqdm

from .. import config
from ..config import refs
from ..exceptions import ConfigurationError
from ..mixins import NotParallel
from ..storage.interfaces import PlacementSet
from .strategy import ConnectionStrategy

if typing.TYPE_CHECKING:
    from ..cell_types import CellType
    from ..storage._files import FileDependencyNode
    from ..topology import Partition


@config.node
class ImportConnectivity(NotParallel, ConnectionStrategy, abc.ABC, classmap_entry=None):
    source: "FileDependencyNode" = config.file(required=True)
    cell_types: list["CellType"] = config.reflist(refs.cell_type_ref, required=False)
    partitions: list["Partition"] = config.reflist(refs.partition_ref, required=False)

    @config.property(default=False)
    def cache(self):
        return self.source.cache

    @cache.setter
    def cache(self, value):
        self.source.cache = bool(value)

    def connect(self, pre, post):
        self.parse_source(pre, post)

    @abc.abstractmethod
    def parse_source(self, pre, post):
        pass


@config.node
class CsvImportConnectivity(ImportConnectivity):
    pre_header: str = config.attr(default="pre")
    post_header: str = config.attr(default="post")
    mapping_key: str = config.attr()
    delimiter: str = config.attr(default=",")
    progress_bar: bool = config.attr(type=bool, default=True)

    def __boot__(self):
        if (
            len(self.presynaptic.cell_types) != 1
            or len(self.postsynaptic.cell_types) != 1
        ):
            raise NotImplementedError(
                "CsvImportConnectivity is strictly from 1 cell type to 1 other cell type"
            )

    def parse_source(self, pre, post):
        pre = pre.placement[0]
        post = post.placement[0]
        if self.mapping_key:

            def make_maps(pre_chunks, post_chunks):
                return self._passover(pre, pre_chunks, post, post_chunks)

            passover_iter = self._load_balance(pre, post)
        else:

            def make_maps(pre_chunks, post_chunks):
                def flush(pre_block, post_block, other_block):
                    with pre.chunk_context(pre_chunks):
                        with post.chunk_context(post_chunks):
                            self.connect_cells(pre, post, pre_block, post_block)

                return lambda x: x, lambda x: x, flush

            passover_iter = ((pre.get_all_chunks(), post.get_all_chunks()),)
        if self.progress_bar:
            passover_iter = tqdm(
                passover_iter, desc="processed", unit="passes", total=len(passover_iter)
            )
        with self.source.provide_stream() as (fp, encoding):
            text = io.TextIOWrapper(fp, encoding=encoding, newline="")
            reader = csv.reader(text)
            headers = [h.strip().lower() for h in next(reader)]
            try:
                cols = [*map(headers.index, (self.pre_header, self.post_header))]
            except ValueError:
                raise ConfigurationError(
                    f"'{self.pre_header}' or '{self.post_header}' not found in "
                    f"'{self.source.file.uri}' column headers {headers}."
                )
            other_cols = [r for r in range(len(headers)) if r not in cols]
            self._other_colnames = [headers[c] for c in other_cols]
            for pre_chunks, post_chunks in passover_iter:
                mappers = make_maps(pre_chunks, post_chunks)
                flush = mappers[2]
                i = 0
                if self.progress_bar:
                    reader = tqdm(reader, desc="imported", unit=" lines")
                pre_block = []
                post_block = []
                other_block = []
                for line in reader:
                    ids = [map_(_safe_int(line[c])) for c, map_ in zip(cols, mappers)]
                    other = [_safe_float(line[c]) for c in other_cols]
                    if ids[0] > -1 and ids[1] > -1:
                        pre_block.append(ids[0])
                        post_block.append(ids[1])
                        other_block.append(other)
                    if i % 100 == 0:
                        est_memsize = (len(other_cols) + 2) * i * 24
                        av_mem = psutil.virtual_memory().available
                        if est_memsize > av_mem / 10:
                            flush(pre_block, post_block, other_block)
                            pre_block = []
                            post_block = []
                            other_block = []
                    i += 1
                flush(pre_block, post_block, other_block)

    def _passover(self, pre: PlacementSet, pre_chunks, post: PlacementSet, post_chunks):
        with pre.chunk_context(pre_chunks):
            data = pre.load_additional(self.mapping_key)
            pre_map = {m: i for i, m in enumerate(data)}.get
        with pre.chunk_context(pre_chunks):
            data = post.load_additional(self.mapping_key)
            post_map = {m: i for i, m in enumerate(data)}.get

        def flush(pre_block, post_block, other_block):
            with pre.chunk_context(pre_chunks):
                with post.chunk_context(post_chunks):
                    self.connect_cells(pre, post, pre_block, post_block)

        return pre_map, post_map, flush

    def _load_balance(self, pre, post, pre_chunks=None, post_chunks=None):
        meminfo = psutil.virtual_memory()
        # Div 16 for array copying safety measures
        if (len(pre) + len(post)) * 24 < meminfo.available / 16:
            return ((pre.get_all_chunks(), post.get_all_chunks()),)
        else:
            raise NotImplementedError(
                "Too many cells to fit into memory. "
                "Piecewise import not implemented yet. Open a GitHub issue."
            )


def _safe_float(value: typing.Any) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _safe_int(value: typing.Any) -> int:
    try:
        return int(float(value))
    except ValueError:
        return -1


__all__ = ["CsvImportConnectivity", "ImportConnectivity"]
