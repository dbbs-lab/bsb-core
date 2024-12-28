import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import shortuuid

from ... import config
from ...services import MPILock
from ..decorators import on_main_until
from ..interfaces import Engine, NoopLock
from ..interfaces import StorageNode as IStorageNode
from .file_store import FileStore


class FileSystemEngine(Engine):
    def __init__(self, root, comm):
        super().__init__(root, comm)
        self._lock = MPILock.sync()
        self._readonly = False

    @property
    def root_slug(self):
        return os.path.relpath(self._root)

    @property
    def versions(self):
        path = Path(self._root) / "versions.txt"
        return json.loads(path.read_text())

    @staticmethod
    def recognizes(root, comm):
        try:
            return os.path.exists(root) and os.path.isdir(root)
        except Exception:
            return False

    def _read(self):
        if self._readonly:
            return NoopLock()
        else:
            return self._lock.read()

    def _write(self):
        if self._readonly:
            raise IOError("Can't perform write operations in readonly mode.")
        else:
            return self._lock.write()

    def _master_write(self):
        if self._readonly:
            raise IOError("Can't perform write operations in readonly mode.")
        else:
            return self._lock.single_write()

    def exists(self):
        return os.path.exists(self._root)

    @on_main_until(lambda self: self.exists())
    def create(self):
        from ... import __version__

        os.makedirs(os.path.join(self._root, "files"), exist_ok=True)
        os.makedirs(os.path.join(self._root, "file_meta"), exist_ok=True)
        path = Path(self._root) / "versions.txt"
        path.write_text(
            json.dumps({"bsb": __version__, "engine": "fs", "version": __version__})
        )

    @on_main_until(lambda self: self.exists())
    def move(self, new_root):
        shutil.move(self._root, new_root)
        self._root = new_root

    @on_main_until(lambda self, r: self.__class__(self.root, self.comm).exists())
    def copy(self, new_root):
        shutil.copytree(self._root, new_root)

    @on_main_until(lambda self: not self.exists())
    def remove(self):
        shutil.rmtree(self._root)

    def require_placement_set(self, ct):
        raise NotImplementedError("No PS")

    def clear_placement(self):
        pass

    def clear_connectivity(self):
        pass

    def get_chunk_stats(self):
        return {}


def _get_default_root():
    return os.path.abspath(
        os.path.join(
            ".",
            "scaffold_network_"
            + datetime.now().strftime("%Y_%m_%d")
            + "_"
            + shortuuid.uuid(),
        )
    )


@config.node
class StorageNode(IStorageNode):
    root = config.attr(type=str, default=_get_default_root, call_default=True)
    """
    Path to the filesystem storage file.
    """
