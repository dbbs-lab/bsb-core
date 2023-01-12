import abc as _abc
import contextlib as _cl
import os
import tempfile
import urllib.parse as _up
import urllib.request as _ur
import pathlib as _pl
import os as _os
import functools as _ft
import typing as _t

from .._util import obj_str_insert

if _t.TYPE_CHECKING:
    from ..core import Scaffold


def _is_uri(url):
    return _up.urlparse(url).scheme != ""


def _uri_to_path(uri):
    parsed = _up.urlparse(uri)
    host = "{0}{0}{mnt}{0}".format(_os.path.sep, mnt=parsed.netloc)
    return _os.path.relpath(
        _os.path.normpath(_os.path.join(host, _ur.url2pathname(_up.unquote(parsed.path))))
    )


class FileDependency:
    def __init__(self, source: str, ext: str = None):
        self._given_source: str = source
        if _is_uri(source):
            self._uri = source
        else:
            self._uri: str = _pl.Path(source).absolute().as_uri()
        self._scheme: "FileScheme" = _get_scheme(_up.urlparse(self._uri).scheme)
        self.file_store: "FileStore" = None
        self.extension = ext

    @obj_str_insert
    def __str__(self):
        return f"'{self._uri}'"

    def get_content(self, check_store=True):
        if check_store:
            stored = self.get_stored_file()
            if stored is None or self._scheme.should_update(self, stored):
                content = self._get_content()
                self.store_content(content)
                return content
            else:
                return stored.load()
        else:
            return self._get_content()

    @_cl.contextmanager
    def provide_locally(self):
        try:
            with self._scheme.provide_locally(self) as path:
                yield path
        except FileNotFoundError:
            if self.file_store is None:
                print("err out")
                raise
            content = self.get_content()
            with tempfile.TemporaryDirectory() as dirpath:
                name = "file"
                if self.extension:
                    name = f"{name}.{self.extension}"
                filepath = _os.path.join(dirpath, name)
                with open(filepath, "wb") as f:
                    f.write(content)
                yield dirpath

    def get_stored_file(self):
        if not self.file_store:
            raise ValueError(
                "Can't check for file dependency in store before scaffold is ready."
            )
        return self.file_store.find_meta("source", self._given_source)

    def store_content(self, content):
        if not self.file_store:
            raise ValueError(
                "Can't store file dependency in store before scaffold is ready."
            )
        return self.file_store.store(content, meta={"source": self._given_source})

    def should_update(self):
        if not self.file_store:
            raise ValueError(
                "Can't update file dependency in store before scaffold is ready."
            )
        stored = self.get_stored_file()
        return stored is None or self._scheme.should_update(self, stored)

    def _get_content(self):
        if not self._scheme.find(self):
            raise FileNotFoundError(f"Couldn't find {self._uri}")
        return self._scheme.get_content(self)

    def update(self, force=False):
        if force or self.should_update():
            self.get_content()


class UriScheme(_abc.ABC):
    @_abc.abstractmethod
    def find(self, file: FileDependency):
        path = _uri_to_path(file._uri)
        return _os.path.exists(path)

    @_abc.abstractmethod
    def should_update(self, file: FileDependency, stored_file):
        path = _uri_to_path(file._uri)
        return _os.path.getmtime(path) > stored_file.mtime

    @_abc.abstractmethod
    def get_content(self, file: FileDependency):
        path = _uri_to_path(file._uri)
        with open(path, "rb") as f:
            return f.read()

    @_abc.abstractmethod
    @_cl.contextmanager
    def provide_locally(self, file: FileDependency):
        if not self.find(file):
            raise FileNotFoundError(f"Can't find {file}.")
        yield _uri_to_path(file._uri)

    def get_local_path(self, file: FileDependency):
        return _uri_to_path(file._uri)


class FileScheme(UriScheme):
    def find(self, file: FileDependency):
        return super().find(file)

    def should_update(self, file: FileDependency, stored_file):
        return super().should_update(file, stored_file)

    def get_content(self, file: FileDependency):
        return super().get_content(file)

    def provide_locally(self, file: FileDependency):
        return super().provide_locally(file)


@_ft.cache
def _get_schemes() -> _t.Mapping[str, FileScheme]:
    from ..plugins import discover

    schemes = discover("storage.schemes")
    schemes["file"] = FileScheme()
    return schemes


def _get_scheme(scheme: str) -> FileScheme:
    schemes = _get_schemes()
    try:
        return schemes[scheme]
    except AttributeError:
        raise AttributeError(f"{scheme} is not a known file scheme.")
