import os
import numpy as np
import functools
from .strategy import ConnectionStrategy
from ..exceptions import SourceQualityError
from .. import config, _util as _gutil
from ..config import types
from ..reporting import warn


@config.node
class Convergence(ConnectionStrategy):
    """
    Connect cells based on a convergence distribution, i.e. by connecting each source cell
    to X target cells.
    """

    convergence = config.attr(type=types.distribution(), required=True)

    def connect(self):
        raise NotImplementedError("Needs to be restored, please open an issue.")


class AllToAll(ConnectionStrategy):
    """
    All to all connectivity between two neural populations
    """

    def get_region_of_interest(self, chunk):
        # All to all needs all pre chunks per post chunk.
        # Fingers crossed for out of memory errors.
        return self._get_all_post_chunks()

    @functools.cache
    def _get_all_post_chunks(self):
        all_ps = (ct.get_placement_set() for ct in self.postsynaptic.cell_types)
        chunks = set(_gutil.ichain(ps.get_all_chunks() for ps in all_ps))
        return list(chunks)

    def connect(self, pre, post):
        for from_ps in pre.placement.values():
            fl = len(from_ps)
            for to_ps in post.placement.values():
                len_ = len(to_ps)
                ml = fl * len_
                src_locs = np.full((ml, 3), -1)
                dest_locs = np.full((ml, 3), -1)
                src_locs[:, 0] = np.repeat(np.arange(fl), len_)
                dest_locs[:, 0] = np.tile(np.arange(len_), fl)
                self.connect_cells(from_ps, to_ps, src_locs, dest_locs)


class ExternalConnections(ConnectionStrategy):
    """
    Load the connection matrix from an external source.
    """

    required = ["source"]
    casts = {"format": str, "warn_missing": bool, "use_map": bool, "headers": bool}
    defaults = {
        "format": "csv",
        "headers": True,
        "use_map": False,
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

    def connect(self):
        if self.format == "csv":
            return self._connect_from_csv()

    def _connect_from_csv(self):
        if not self.check_external_source():
            src = self.get_external_source()
            raise RuntimeError(f"Missing source file '{src}' for `{self.name}`.")
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        # Read the entire csv, skipping the headers if there are any.
        data = np.loadtxt(
            self.get_external_source(),
            skiprows=int(self.headers),
            delimiter=self.delimiter,
        )
        if self.use_map:

            def emap_name(t):
                return t.placement.name + "_ext_map"

            from_gid_map = self.scaffold.load_appendix(emap_name(from_type))
            to_gid_map = self.scaffold.load_appendix(emap_name(to_type))
            from_targets = self.scaffold.get_placement_set(from_type).identifiers
            to_targets = self.scaffold.get_placement_set(to_type).identifiers
            data[:, 0] = self._map(data[:, 0], from_gid_map, from_targets)
            data[:, 1] = self._map(data[:, 1], to_gid_map, to_targets)
        self.scaffold.connect_cells(self, data)

    def _map(self, data, map, targets):
        # Create a dict with pairs between the map and the target values
        # Vectorize its dictionary lookup and perform the vector function on the data
        try:
            return np.vectorize(dict(zip(map, targets)).get)(data)
        except TypeError:
            raise SourceQualityError("Missing GIDs in external map.")
