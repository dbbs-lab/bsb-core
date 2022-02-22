import os
import io
import pathlib


class _NoLink:
    def exists(self):
        return False

    def get(self):
        return io.StringIO("")


class FileLink:
    def __init__(self, source, id, store=None, update="always", binary=False):
        self._src = source
        if source not in ("sys", "store"):
            raise ValueError(f"'{source}' not a valid link source. Pick 'sys' or 'store'")
        if source == "store" and store is None:
            raise ValueError("`store` argument required for filestore links.")
        elif source == "sys" and os.path.abspath(id) != str(id):
            raise ValueError("Filesystem links must be absolute")
        self.store = store
        self.id = id
        self._upd = update
        self._b = binary

    def __str__(self):
        return (
            "<"
            + ("filesystem" if self._src == "sys" else "file store")
            + f" link '{self.id}'>"
        )

    def exists(self):
        if self._src == "store":
            return self.id in self.store.all()
        else:
            return os.path.exists(self.id)

    def get(self, binary=None):
        binary = self._b if binary is None else binary
        if self._src == "sys":
            return open(self.id, f"r{'b' if binary else ''}")
        else:
            return self.store.stream(self.id, binary=binary)


def syslink(path, update="always"):
    return FileLink("sys", pathlib.Path(os.path.abspath(path)), update=update)


def storelink(store, id, update="always"):
    return FileLink("store", id, store=store, update=update)


def link(store, proj_dir, source, id, update):
    if source == "sys":
        return FileLink("sys", proj_dir / id, update=update)
    elif source == "store":
        return FileLink("store", id, store=store, update=update)
    else:
        raise ValueError(f"'{source}' not a valid link source. Pick 'sys' or 'store'")


def nolink():
    return _NoLink()
