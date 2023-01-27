import abc as _abc
import contextlib as _cl
import datetime as _dt
import tempfile as _tf
import time as _t
import urllib.parse as _up
import urllib.request as _ur
import pathlib as _pl
import os as _os
import functools as _ft
import typing as _tp
import requests as _rq
import email.utils as _eml
import nrrd as _nrrd

from .._util import obj_str_insert
from .. import config
from ..config import types

if _tp.TYPE_CHECKING:
    from ..storage.interfaces import FileStore
    from ..morphologies import Morphology


def _is_uri(url):
    return _up.urlparse(url).scheme != ""


def _uri_to_path(uri):
    parsed = _up.urlparse(uri)
    host = "{0}{0}{mnt}{0}".format(_os.path.sep, mnt=parsed.netloc)
    return _os.path.relpath(
        _os.path.normpath(_os.path.join(host, _ur.url2pathname(_up.unquote(parsed.path))))
    )


class FileDependency:
    def __init__(
        self,
        source: str,
        file_store: "FileStore" = None,
        ext: str = None,
        cache=True,
    ):
        self._given_source: str = source
        if _is_uri(source):
            self._uri = source
        else:
            self._uri: str = _pl.Path(source).absolute().as_uri()
        self._scheme: "FileScheme" = _get_scheme(_up.urlparse(self._uri).scheme)
        self.file_store = file_store
        self.extension = ext
        self.cache = cache

    @property
    def uri(self):
        return self._uri

    @obj_str_insert
    def __str__(self):
        return f"'{self._uri}'"

    def get_content(self, check_store=True):
        if check_store:
            stored = self.get_stored_file()
            if stored is None or self._scheme.should_update(self, stored):
                content = self._get_content()
                if self.cache:
                    self.store_content(content, meta=self._scheme.get_meta(self))
                return content
            else:
                return stored.load()
        else:
            return self._get_content()

    def get_meta(self, check_store=True):
        if (
            not check_store or ((stored := self.get_stored_file()) is None)
        ) or self._scheme.should_update(self, stored):
            return self._scheme.get_meta(self)
        else:
            return stored.meta

    @_cl.contextmanager
    def provide_locally(self):
        try:
            path = self._scheme.get_local_path(self)
            if _os.path.exists(path):
                yield (path, None)
            else:
                raise FileNotFoundError()
        except (TypeError, FileNotFoundError):
            if self.file_store is None:
                raise FileNotFoundError(f"Can't find {self}")
            content = self.get_content()
            with _tf.TemporaryDirectory() as dirpath:
                name = "file"
                if self.extension:
                    name = f"{name}.{self.extension}"
                filepath = _os.path.join(dirpath, name)
                with open(filepath, "wb") as f:
                    f.write(content[0])
                yield (filepath, content[1])

    def provide_stream(self):
        return self._scheme.provide_stream(self)

    def get_stored_file(self):
        if not self.file_store:
            raise ValueError(
                "Can't check for file dependency in store before scaffold is ready."
            )
        return self.file_store.find_meta("source", self._given_source)

    def store_content(self, content, encoding=None, meta=None):
        if not self.file_store:
            raise ValueError(
                "Can't store file dependency in store before scaffold is ready."
            )
        if isinstance(content, tuple):
            content, encoding = content
        # Save the file under the same id if it already exists
        id_ = getattr(self.get_stored_file(), "id", None)
        if meta is None:
            meta = {}
        meta.update(self._scheme.get_meta(self))
        meta["source"] = self._given_source
        return self.file_store.store(
            content, meta=meta, encoding=encoding, id=id_, overwrite=True
        )

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
        path = _uri_to_path(file.uri)
        return _os.path.exists(path)

    @_abc.abstractmethod
    def should_update(self, file: FileDependency, stored_file):
        path = _uri_to_path(file.uri)
        return _os.path.getmtime(path) > stored_file.mtime

    @_abc.abstractmethod
    def get_content(self, file: FileDependency):
        with self.provide_stream(file) as (fp, encoding):
            fp: _tp.BinaryIO
            return (fp.read(), encoding)

    @_cl.contextmanager
    @_abc.abstractmethod
    def provide_stream(self, file):
        path = _uri_to_path(file.uri)
        with open(path, "rb") as fp:
            yield (fp, None)

    @_abc.abstractmethod
    def get_meta(self, file: FileDependency):
        return {}

    @_abc.abstractmethod
    def get_local_path(self, file: FileDependency):
        raise RuntimeError(f"{type(self)} has no local path representation")


class FileScheme(UriScheme):
    def find(self, file: FileDependency):
        return super().find(file)

    def should_update(self, file: FileDependency, stored_file):
        return super().should_update(file, stored_file)

    def get_content(self, file: FileDependency):
        return super().get_content(file)

    def provide_stream(self, file: FileDependency):
        return super().provide_stream(file)

    def get_meta(self, file: FileDependency):
        return super().get_meta(file)

    def get_local_path(self, file: FileDependency):
        return _uri_to_path(file.uri)


class UrlScheme(UriScheme):
    def find(self, file: FileDependency):
        response = _rq.head(file.uri)
        return response.status_code == 200

    def should_update(self, file: FileDependency, stored_file):
        mtime = stored_file.mtime
        headers = self.get_meta(file).get("headers", {})
        stored_headers = stored_file.meta.get("headers", {})
        if "ETag" in headers:
            new_etag = headers["ETag"]
            old_etag = stored_headers.get("ETag")
            # Check if we have the latest ETag
            return old_etag != new_etag
        elif "Last-Modified" in headers:
            their_mtime = _dt.datetime(
                *_eml.parsedate(headers["Last-Modified"])[:6]
            ).timestamp()
            return their_mtime > mtime
        # 100h default expiration
        return _t.time() > mtime + 360000

    def get_content(self, file: FileDependency):
        response = _rq.get(file.uri)
        return (response.content, response.encoding)

    def get_meta(self, file: FileDependency):
        response = _rq.head(file.uri)
        return {"headers": dict(response.headers)}

    def get_local_path(self, file: FileDependency):
        raise TypeError("URL schemes don't have a local path")

    @_cl.contextmanager
    def provide_stream(self, file):
        response = _rq.get(file.uri, stream=True)
        response.raw.decode_content = True
        response.raw.auto_close = False
        yield (response.raw, response.encoding)


@_ft.cache
def _get_schemes() -> _tp.Mapping[str, FileScheme]:
    from ..plugins import discover

    schemes = discover("storage.schemes")
    schemes["file"] = FileScheme()
    schemes["http"] = schemes["https"] = UrlScheme()
    return schemes


def _get_scheme(scheme: str) -> FileScheme:
    schemes = _get_schemes()
    try:
        return schemes[scheme]
    except AttributeError:
        raise AttributeError(f"{scheme} is not a known file scheme.")


@config.node
class FileDependencyNode:
    file: "FileDependency" = config.attr(type=FileDependency)

    def __init__(self, value=None, **kwargs):
        if value is not None:
            self.file = value

    def __boot__(self):
        self.file.file_store = self.scaffold.files
        self.file.update()

    def __inv__(self):
        if not isinstance(self, FileDependencyNode):
            return self
        if self._config_pos_init:
            return self.file._given_source
        else:
            return self.__tree__()

    def load_object(self):
        return self.file.get_content()

    def provide_locally(self):
        return self.file.provide_locally()

    def provide_stream(self):
        return self.file.provide_stream()


@config.node
class CodeDependencyNode(FileDependencyNode):
    module: str = config.attr(type=str)

    @config.property
    def file(self):
        if getattr(self, "scaffold", None) is not None:
            file_store = self.scaffold.files
        else:
            file_store = None
        return FileDependency(
            self.module.replace(".", _os.sep) + ".py", file_store=file_store
        )

    def __init__(self, module=None, **kwargs):
        super().__init__(**kwargs)
        if module is not None:
            self.module = module

    def load_object(self):
        import importlib.util
        import sys

        sys.path.append(_os.getcwd())
        try:
            with self.file.provide_locally() as (path, encoding):
                spec = importlib.util.spec_from_file_location(self.module, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[self.module] = module
                spec.loader.exec_module(module)
        finally:
            tmp = list(reversed(sys.path))
            tmp.remove(_os.getcwd())
            sys.path = list(reversed(tmp))


@config.node
class Operation:
    func = config.attr(type=types.function_())
    parameters = config.catch_all()

    def __init__(self, value=None, /, **kwargs):
        if value is not None:
            self.func = value

    def process(self, obj):
        return self.func(obj, self.parameters)


class FilePipelineMixin:
    pipeline = config.list(type=Operation)

    def pipe(self, input):
        return _ft.reduce(lambda state, func: func(state), self.pipeline, input)


@config.node
class NrrdDependencyNode(FilePipelineMixin, FileDependencyNode):
    def get_header(self):
        with self.file.provide_locally() as (path, encoding):
            return _nrrd.read_header(path)

    def get_data(self):
        with self.file.provide_locally() as (path, encoding):
            return _nrrd.read(path)[0]

    def load_object(self):
        return self.pipe(self.get_data())


@config.node
class MorphologyDependencyNode(FilePipelineMixin, FileDependencyNode):
    def load_object(self) -> "Morphology":
        from ..morphologies import Morphology

        with self.file.provide_locally() as (path, encoding):
            return self.pipe(Morphology.from_file(path))
