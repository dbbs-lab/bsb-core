import base64
import json
import os
import time
import typing
from uuid import uuid4

from ...exceptions import MissingActiveConfigError
from ..interfaces import FileStore as IFileStore


def _id_to_path(url):
    return base64.urlsafe_b64encode(url.encode("UTF-8")).decode("UTF-8")


def _path_to_id(f):
    return base64.urlsafe_b64decode(f).decode("UTF-8")


class FileStore(IFileStore):
    def file_path(self, *paths):
        return os.path.join(self._engine.root, "files", *paths)

    def id_to_file_path(self, id):
        path = _id_to_path(id)
        return self.file_path(path)

    def meta_path(self, *paths):
        return os.path.join(self._engine.root, "file_meta", *paths)

    def id_to_meta_path(self, id):
        path = _id_to_path(id)
        return self.meta_path(path)

    def all(self):
        return {
            id: self.get_meta(id) for id in map(_path_to_id, os.listdir(self.file_path()))
        }

    def store(self, content, id=None, meta=None, encoding=None, overwrite=False):
        if isinstance(content, str):
            if encoding is None:
                encoding = "utf-8"
            content = content.encode(encoding)
        if id is None:
            id = str(uuid4())
        if meta is None:
            meta = {}
        if not overwrite and self.has(id):
            raise FileExistsError(f"Store already contains a file with id {id}")
        with open(self.id_to_file_path(id), "wb") as f:
            f.write(content)
        with open(self.id_to_meta_path(id), "w") as f:
            json.dump({"meta": meta, "mtime": time.time(), "encoding": encoding}, f)
        return id

    def load(self, id):
        """
        Load the content of an object in the file store.

        :param id: id of the content to be loaded.
        :type id: str
        :returns: The content of the stored object
        :rtype: str
        :raises FileNotFoundError: The given id doesn't exist in the file store.
        """
        with open(self.id_to_file_path(id), "rb") as f:
            content = f.read()
        our_meta = self._get_meta(id)
        meta = our_meta["meta"]
        encoding = our_meta["encoding"]
        return (content.decode(encoding) if encoding else content), meta

    def remove(self, id):
        """
        Remove the content of an object in the file store.

        :param id: id of the content to be removed.
        :type id: str
        :raises FileNotFoundError: The given id doesn't exist in the file store.
        """
        os.unlink(self.id_to_file_path(id))
        os.unlink(self.id_to_meta_path(id))

    def store_active_config(self, config):
        """
        Store configuration in the file store and mark it as the active configuration of
        the stored network.

        :param config: Configuration to be stored
        :type config: :class:`~.config.Configuration`
        :returns: The id the config was stored under
        :rtype: str
        """
        return self.store(json.dumps(config.__tree__()), meta={"active_config": True})

    def load_active_config(self):
        """
        Load the active configuration stored in the file store.

        :returns: The active configuration
        :rtype: :class:`~.config.Configuration`
        :raises Exception: When there's no active configuration in the file store.
        """
        from ...config import Configuration

        stored = self.find_meta("active_config", True)
        if stored is None:
            raise MissingActiveConfigError("No active config")
        else:
            content, meta = stored.load()
            tree = json.loads(content)
            cfg = Configuration(**tree, _store=self)
            cfg._meta = meta
            return cfg

    def has(self, id):
        """
        Must return whether the file store has a file with the given id.
        """
        return os.path.exists(self.id_to_file_path(id))

    def get_mtime(self, id):
        """
        Must return the last modified timestamp of file with the given id.
        """
        return self._get_meta(id)["mtime"]

    def get_encoding(self, id):
        """
        Must return the encoding of the file with the given id, or None if it is
        unspecified binary data.
        """
        return self._get_meta(id)["encoding"]

    def _get_meta(self, id) -> typing.Mapping[str, typing.Any]:
        """
        Must return the metadata of the given id.
        """
        with open(self.id_to_meta_path(id), "r") as f:
            return json.load(f)

    def get_meta(self, id) -> typing.Mapping[str, typing.Any]:
        """
        Must return the metadata of the given id.
        """
        return self._get_meta(id)["meta"]
