from .... import config
from ....config.nodes import StorageNode as IStorageNode
from ...interfaces import Engine
from contextlib import contextmanager
from .placement_set import PlacementSet
from .connectivity_set import ConnectivitySet
from .file_store import FileStore
from .morphology_repository import MorphologyRepository
from datetime import datetime
import h5py
import os
from mpilock import sync


class HDF5Engine(Engine):
    def __init__(self, root):
        super().__init__(root)
        self._root = root
        self._lock = sync()

    def __eq__(self, other):
        return self._format == getattr(other, "_format", None) and self._root == getattr(
            other, "_root", None
        )

    def _read(self):
        return self._lock.read()

    def _write(self):
        return self._lock.write()

    def _master_write(self):
        return self._lock.single_write()

    def _handle(self, mode):
        return h5py.File(self._root, mode)

    def exists(self):
        return os.path.exists(self._root)

    def create(self):
        with self._write():
            with self._handle("w") as handle:
                handle.create_group("cells")
                handle.create_group("placement")
                handle.create_group("connectivity")
                handle.create_group("files")
                handle.create_group("morphologies")

    def move(self, new_root):
        from shutil import move

        with self._write():
            move(self._root, new_root)

        self._root = new_root

    def remove(self):
        with self._write() as fence:
            os.remove(self._root)

    def clear_placement(self):
        with self._write():
            with self._handle("a") as handle:
                handle.require_group("placement")
                del handle["placement"]
                handle.require_group("placement")

    def clear_connectivity(self):
        with self._write():
            with self._handle("a") as handle:
                handle.require_group("connectivity")
                del handle["connectivity"]
                handle.require_group("connectivity")


def _get_default_root():
    return os.path.abspath(
        os.path.join(
            ".",
            "scaffold_network_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".hdf5",
        )
    )


@config.node
class StorageNode(IStorageNode):
    root = config.attr(type=str, default=_get_default_root, call_default=True)
