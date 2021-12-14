from ....exceptions import *
from .resource import Resource
from ...interfaces import ConnectivitySet as IConnectivitySet
from .chunks import get_chunk_tag
import numpy as np

_root = "/cells/connections/"

class ConnectivitySet(
    Resource,
    IConnectivitySet
):
    """
    Fetches placement data from storage.

    .. note::

        Use :func:`.core.get_connectivity_set` to correctly obtain a
        :class:`ConnectivitySet <.storage.interfaces.IConnectivitySet>`.
    """

    def __init__(self, engine, tag):
        self.tag = tag
        super().__init__(engine, _root + tag)
        if not self.exists(engine, tag):
            raise DatasetNotFoundError("ConnectivitySet '{}' does not exist".format(tag))

    @classmethod
    def create(cls, engine, tag):
        """
        Create the structure for this connectivity set in the HDF5 file. Connectivity sets are
        stored under ``/cells/connections/<tag>``.
        """
        path = _root + tag
        with engine._write() as fence:
            with engine._handle("a") as h:
                g = h.create_group(path)
        return cls(engine, tag)

    @staticmethod
    def exists(engine, tag):
        with engine._read():
            with engine._handle("r") as h:
                return _root + tag in h

    @classmethod
    def require(cls, engine, tag):
        path = _root + tag
        with engine._write():
            with engine._handle("a") as h:
                g = h.require_group(path)
        return cls(engine, tag)

    def append_data(self, src_chunk, dest_chunk, src_locs, dest_locs):
        if len(src_locs) != len(dest_locs):
            raise ValueError("Connectivity matrices must be of same length.")
        with self._engine._write():
            with self._engine._handle("a") as h:
                self._append_data(src_chunk, dest_chunk, src_locs, dest_locs, h)

    def _append_data(self, src_chunk, dest_chunk, src_locs, dest_locs, h):
        # Require the data
        grp = h.require_group(self._path + get_chunk_tag(dest_chunk))
        unpack_me = [None, None]
        for i, tag in enumerate(("source_loc", "dest_loc")):
            if tag in grp:
                unpack_me[i] = grp[tag]
            else:
                # Require dataset is absolutely useless as existing shape must be known ...
                unpack_me[i] = grp.create_dataset(tag, shape=(0, 3), dtype=int, chunks=(1024, 3), maxshape=(None, 3))
        src_ds, dest_ds = unpack_me
        # Move the pointers that keep track of the chunks
        new_rows = len(src_locs)
        total = len(src_ds)
        print("Inserting", new_rows, "rows")
        print("Total data length", total)
        self._store_pointers(grp, src_chunk, new_rows, total)
        iptr, eptr = self._get_insert_pointers(grp, src_chunk)
        print("Stored data from line", iptr, "to", eptr)
        if eptr is None:
            eptr = total + new_rows
        # Resize and insert data.
        src_ds.resize(len(src_ds) + new_rows, axis=0)
        dest_ds.resize(len(dest_ds) + new_rows, axis=0)
        src_ds[iptr:eptr] = np.concatenate((src_ds[iptr:(eptr - new_rows)], src_locs))
        dest_ds[iptr:eptr] = np.concatenate((dest_ds[iptr:(eptr - new_rows)], dest_locs))

    def _store_pointers(self, group, chunk, n, total):
        print("Storing pointers", chunk, n)
        cname = get_chunk_tag(chunk)
        chunks = list(tuple(t) for t in group.attrs.get("chunk_list", []))
        print("Chunks already present:", chunks)
        if chunk in chunks:
            print("Just moving up pointers")
            # Source chunk already existed, just increment the subseq. pointers
            inc_from = chunks.index(chunk) + 1
        else:
            print("Appending new chunk to end")
            # We are the last chunk, we start adding rows at the end.
            group.attrs[cname] = total
            # Move up the increment pointer to place ourselves after the
            # last element
            chunks.append(chunk)
            inc_from = len(chunks)
            group.attrs["chunk_list"] = chunks
        # Increment the pointers of the chunks that follow us, by `n` rows.
        for c in chunks[inc_from:]:
            group.attrs[get_chunk_tag(c)] += n

    def _get_insert_pointers(self, group, chunk):
        cname = get_chunk_tag(chunk)
        chunks = list(tuple(t) for t in group.attrs["chunk_list"])
        iptr = group.attrs[cname]
        idx = chunks.index(chunk)
        if idx + 1 == len(chunks):
            # Last chunk, no end pointer
            eptr = None
        else:
            # Get the pointer of the next chunk
            eptr = group.attrs[get_chunk_tag(chunks[idx + 1])]
        return iptr, eptr

    def _get_chunk_data(self, dest_chunk):
        with self._engine._write():
            with self._engine._handle("a") as h:
                grp = h[self._path + get_chunk_tag(dest_chunk)]
                src_chunks = grp.attrs["chunk_list"]
                chunk_ptrs = [grp.attrs[get_chunk_tag(tuple(c))] for c in src_chunks]
                src = grp["source_loc"][()]
                dest = grp["dest_loc"][()]
        return src_chunks, chunk_ptrs, src, dest

def _sort_triple(a, b):
    # Comparator for chunks by bitshift and sum of the coords.
    return (a[0] << 42 + a[1] << 21 + a[2]) > (b[0] << 42 + b[1] << 21 + b[2])
