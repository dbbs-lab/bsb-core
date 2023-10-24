import typing
import warnings

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
import hashlib as _hl
import yaml

from .._util import obj_str_insert
from .. import config
from ..config import types
from ..config._attrs import cfglist

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

    def __hash__(self):
        return hash(self._uri)

    def __inv__(self):
        return self._given_source

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
        try:
            file_mtime = _os.path.getmtime(path)
        except FileNotFoundError:
            return False
        try:
            stored_mtime = stored_file.mtime
        except Exception:
            return True
        return file_mtime > stored_mtime

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
    def resolve_uri(self, file: FileDependency):
        return file.uri

    def find(self, file: FileDependency):
        with self.create_session() as session:
            response = session.head(self.resolve_uri(file))
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
        with self.create_session() as session:
            response = session.get(self.resolve_uri(file))
        return (response.content, response.encoding)

    def get_meta(self, file: FileDependency):
        with self.create_session() as session:
            response = session.head(self.resolve_uri(file))
        return {"headers": dict(response.headers)}

    def get_local_path(self, file: FileDependency):
        raise TypeError("URL schemes don't have a local path")

    @_cl.contextmanager
    def provide_stream(self, file):
        with self.create_session() as session:
            response = session.get(self.resolve_uri(file), stream=True)
        response.raw.decode_content = True
        response.raw.auto_close = False
        yield (response.raw, response.encoding)

    def create_session(self):
        return _rq.Session()


class NeuroMorphoScheme(UrlScheme):
    _nm_url = "https://neuromorpho.org/"
    _meta = "api/neuron/name/"
    _files = "dableFiles/"

    def resolve_uri(self, file: FileDependency):
        meta = self.get_nm_meta(file)
        return self._swc_url(meta["archive"], meta["neuron_name"])

    @_ft.cache
    def get_nm_meta(self, file: FileDependency):
        name = _up.urlparse(file.uri).hostname
        # urlparse gives lowercase, so slice out the original cased NM name
        idx = file.uri.lower().find(name)
        name = file.uri[idx : (idx + len(name))]
        try:
            with self.create_session() as session:
                res = session.get(self._nm_url + self._meta + name)
        except Exception as e:
            return {"archive": "none", "neuron_name": "none"}
        if res.status_code == 404:
            raise IOError(f"'{name}' is not a valid NeuroMorpho name.")
        elif res.status_code != 200:
            raise IOError("NeuroMorpho API error: " + res.text)
        return res.json()

    def get_meta(self, file: FileDependency):
        meta = super().get_meta(file)
        meta["neuromorpho_data"] = self.get_nm_meta(file)
        return meta

    @classmethod
    def _swc_url(cls, archive, name):
        base_url = f"{cls._nm_url}{cls._files}{_up.quote(archive.lower())}"
        return f"{base_url}/CNG%20version/{name}.CNG.swc"

    def create_session(self):
        # Weak DH key on neuromorpho.org
        # https://stackoverflow.com/a/76217135/1016004
        from requests.adapters import HTTPAdapter
        from urllib3.util import create_urllib3_context
        from urllib3 import PoolManager

        class DHAdapter(HTTPAdapter):
            def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
                ctx = create_urllib3_context(ciphers=":HIGH:!DH:!aNULL")
                self.poolmanager = PoolManager(
                    num_pools=connections,
                    maxsize=maxsize,
                    block=block,
                    ssl_context=ctx,
                    **kwargs,
                )

        session = _rq.Session()
        session.mount(self._nm_url, DHAdapter())
        return session


@_ft.cache
def _get_schemes() -> _tp.Mapping[str, FileScheme]:
    from ..plugins import discover

    schemes = discover("storage.schemes")
    schemes["file"] = FileScheme()
    schemes["http"] = schemes["https"] = UrlScheme()
    schemes["nm"] = NeuroMorphoScheme()
    return schemes


def _get_scheme(scheme: str) -> FileScheme:
    schemes = _get_schemes()
    try:
        return schemes[scheme]
    except KeyError:
        raise KeyError(f"{scheme} is not a known file scheme.")


@config.node
class FileDependencyNode:
    file: "FileDependency" = config.attr(type=FileDependency)

    def __init__(self, value=None, **kwargs):
        if value is not None:
            self.file = value

    def __boot__(self):
        self.file.file_store = self.scaffold.files
        if self.scaffold.is_main_process():
            self.file.update()

    def __inv__(self):
        if not isinstance(self, FileDependencyNode):
            return self
        if self._config_pos_init:
            return self.file._given_source
        else:
            tree = self.__tree__()
            return tree

    def load_object(self):
        return self.file.get_content()

    def provide_locally(self):
        return self.file.provide_locally()

    def provide_stream(self):
        return self.file.provide_stream()

    def get_stored_file(self):
        return self.file.get_stored_file()


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


class OperationCallable(typing.Protocol):
    def __call__(self, obj: object, **kwargs: typing.Any) -> object:
        pass


@config.node
class Operation:
    func: OperationCallable = config.attr(type=types.function_())
    parameters: dict[typing.Any] = config.catch_all(type=types.any_())

    def __init__(self, value=None, /, **kwargs):
        if value is not None:
            self.func = value

    def __call__(self, obj):
        return self.func(obj, **self.parameters)


class FilePipelineMixin:
    pipeline: cfglist[Operation] = config.list(type=Operation)

    def pipe(self, input):
        return _ft.reduce(lambda state, func: func(state), self.pipeline, input)


@config.node
class NrrdDependencyNode(FilePipelineMixin, FileDependencyNode):
    """
    Configuration dependency node to load NRRD files.
    """

    def get_header(self):
        with self.file.provide_locally() as (path, encoding):
            return _nrrd.read_header(path)

    def get_data(self):
        with self.file.provide_locally() as (path, encoding):
            return _nrrd.read(path)[0]

    def load_object(self):
        return self.pipe(self.get_data())


class MorphologyOperationCallable(OperationCallable):
    def __call__(self, obj: "Morphology", **kwargs: typing.Any) -> "Morphology":
        pass


@config.node
class MorphologyOperation(Operation):
    func: MorphologyOperationCallable = config.attr(
        type=types.method_shortcut("bsb.morphologies.Morphology")
    )


@config.node
class MorphologyDependencyNode(FilePipelineMixin, FileDependencyNode):
    """
    Configuration dependency node to load morphology files.
    The content of these files will be stored in bsb.morphologies.Morphology instances.
    """

    pipeline: cfglist[MorphologyOperation] = config.list(type=MorphologyOperation)
    name: str = config.attr(type=str, default=None, required=False)
    """
    Name associated to the morphology. If not provided, the program will use the name of the file 
    in which the morphology is stored. 
    """
    tags: dict[typing.Union[str, list[str]]] = config.attr(
        type=types.dict(type=types.or_(types.str(), types.list(str)))
    )
    """
    Dictionary mapping SWC tags to sets of morphology labels.
    """

    def store_content(self, content, *args, encoding=None, meta=None):
        if meta is None:
            meta = {}
        meta["_hash"] = self._hash(content)
        meta["_stale"] = True
        stored = super().store_content(content, *args, encoding=encoding, meta=meta)
        return stored

    def load_object(self) -> "Morphology":
        from ..morphologies import Morphology

        self.file.update()
        stored = self.get_stored_file()
        meta = stored.meta
        if meta.get("_stale", True):
            content, meta = stored.load()
            try:
                morpho_in = Morphology.from_buffer(content, meta=meta)
            except Exception as _:
                with self.file.provide_locally() as (path, encoding):
                    morpho_in = Morphology.from_file(path, tags=self.tags, meta=meta)
            morpho = self.pipe(morpho_in)
            meta["hash"] = self._hash(content)
            meta["_stale"] = False
            morpho.meta = meta
            self.scaffold.morphologies.save(
                self.get_morphology_name(), morpho, overwrite=True
            )
            return morpho
        else:
            return self.scaffold.morphologies.load(self.get_morphology_name())

    def get_morphology_name(self):
        """
        Returns morphology name provided by the user or extract it from its file name.

        :returns: Morphology name
        :rtype: str
        """
        return self.name if self.name is not None else _pl.Path(self.file.uri).stem

    def store_object(self, morpho, hash_):
        """
        Save a morphology into the circuit file under the name of this instance morphology.

        :param hash_: Hash key to store as metadata with the morphology
        :type hash_: str
        :param morpho: Morphology to store
        :type morpho: bsb.morphologies.Morphology
        """
        self.scaffold.morphologies.save(
            self.get_morphology_name(), morpho, meta={"hash": hash_}
        )

    def _hash(self, content):
        md5 = _hl.new("md5", usedforsecurity=False)
        if isinstance(content, str):
            md5.update(content.encode("utf-8"))
        else:
            md5.update(content)
        return md5.hexdigest()

    def queue(self, pool):
        """
        Add the loading of the current morphology to a job queue.

        :param pool: Queue of jobs.
        :type pool:bsb.services.pool.JobPool
        """
        pool.queue(
            lambda scaffold, i=self._config_index: scaffold.configuration.morphologies[
                i
            ].load_object()
        )


@config.node
class YamlDependencyNode(FileDependencyNode):
    """
    Configuration dependency node to load yaml files.
    """

    def load_object(self):
        with self.file.provide_locally() as (path, encoding):
            return yaml.safe_load(open(path, "r"))
