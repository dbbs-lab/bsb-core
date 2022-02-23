import os
import io
import pathlib
import appdirs
import contextlib

_bsb_dirs = appdirs.AppDirs("bsb")
_cache_path = pathlib.Path(_bsb_dirs.user_cache_dir)


class _NoLink:
    def exists(self):
        return False

    def get(self):
        return io.StringIO("")


class FileLink:
    _type = "sys"

    def __init__(self, path, update="always", binary=False):
        self.path = pathlib.Path(path).resolve()
        self.upd_mode = update
        self.binary = binary

    @property
    def type(self):
        return self._type

    def __str__(self):
        return f"<filesystem link '{self.path}'>"

    def exists(self):
        return self.path.exists()

    def should_update(self, last_retrieved=None):
        if not self.exists():
            return False
        if self.upd_mode == "never":
            return False
        if last_retrieved is None:
            return True
        else:
            return os.path.getmtime(self.path) > last_retrieved

    def get(self, binary=None):
        binary = self.binary if binary is None else binary
        return open(self.path, f"r{'b' if binary else ''}")

    def set(self, binary=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        binary = self.binary if binary is None else binary
        return open(self.path, f"w{'b' if binary else ''}")


class CacheLink(FileLink):
    _type = "cache"

    def __init__(self, path, update="never", binary=False):
        path = _cache_path / path
        super().__init__(path, update=update, binary=binary)

    def should_update(self, source_timestamp=None):
        if not self.exists():
            return True
        if self.upd_mode == "never":
            return False
        return os.path.getmtime(self.path) < source_timestamp


class StoreLink(FileLink):
    _type = "store"

    def __init__(self, store, id, update="always", binary=False):
        self.store = store
        self.id = id
        self.upd_mode = update
        self.binary = binary

    def __str__(self):
        return f"<filestore link '{self.id}@{self.store}'>"

    def exists(self):
        return self.id in self.store.all()

    def should_update(self):
        return False

    def get(self, binary=None):
        binary = self.binary if binary is None else binary
        return self.store.stream(self.id, binary=binary)


def syslink(path, update="always", binary=False):
    return FileLink(pathlib.Path(os.path.abspath(path)), update=update, binary=binary)


def cachelink(path, update="never", binary=False):
    return CacheLink(path, update=update, binary=binary)


def storelink(store, id, update="always", binary=False):
    return StoreLink(store, id, update=update, binary=binary)


def link(store, proj_dir, source, id, update):
    if source == "sys":
        return FileLink(proj_dir / id, update=update)
    elif source == "store":
        return StoreLink(store, id, update=update)
    elif source == "cache":
        return CacheLink(id, update=update)
    else:
        raise ValueError(
            f"'{source}' not a valid link source. Pick 'sys', 'store' or 'cache'."
        )


def nolink():
    return _NoLink()
