import abc
import concurrent
import re
import tempfile
import typing
import urllib
import warnings
from concurrent.futures import ThreadPoolExecutor

import requests

from .. import config
from ..config import types
from ..config._attrs import cfglist
from ..exceptions import MissingMorphologyError, SelectorError
from . import Morphology

if typing.TYPE_CHECKING:
    from ..core import Scaffold


@config.dynamic(
    attr_name="select",
    auto_classmap=True,
    required=False,
    default="by_name",
)
class MorphologySelector(abc.ABC):
    scaffold: "Scaffold"

    @abc.abstractmethod
    def validate(self, all_morphos):
        pass

    @abc.abstractmethod
    def pick(self, morphology):
        pass


@config.node
class NameSelector(MorphologySelector, classmap_entry="by_name"):
    names: cfglist[str] = config.list(type=str, required=types.shortform())

    def __init__(self, name=None, /, **kwargs):
        if name is not None:
            self.names = [name]

    def __inv__(self):
        if self._config_pos_init:
            return self.names[0]
        return self.__tree__()

    def _cache_patterns(self):
        self._pnames = {n: n.replace("*", r".*").replace("|", "\\|") for n in self.names}
        self._patterns = {n: re.compile(f"^{pat}$") for n, pat in self._pnames.items()}
        self._empty = not self.names
        self._match = re.compile(f"^({'|'.join(self._pnames.values())})$")

    def validate(self, all_morphos):
        self._cache_patterns()
        repo_names = {m.get_meta()["name"] for m in all_morphos}
        missing = [
            n
            for n, pat in self._patterns.items()
            if not any(pat.match(rn) for rn in repo_names)
        ]
        if missing:
            err = "Morphology repository misses the following morphologies"
            if self._config_parent is not None:
                node = self._config_parent._config_parent
                err += f" required by {node.get_node_name()}"
            err += f": {', '.join(missing)}"
            raise MissingMorphologyError(err)

    def pick(self, morphology):
        self._cache_patterns()
        return (
            not self._empty
            and self._match.match(morphology.get_meta()["name"]) is not None
        )


@config.node
class NeuroMorphoSelector(NameSelector, classmap_entry="from_neuromorpho"):
    _url = "http://cng.gmu.edu:8080/neuroMorpho/"  # "https://neuromorpho.org/"
    _meta = "api/neuron/select?q=neuron_name:"
    _files = "dableFiles/"

    def __boot__(self):
        if self.scaffold.is_main_process():
            try:
                morphos = self._scrape_nm(self.names)
            except:
                self.scaffold._comm.barrier()
                raise
            for name, morpho in morphos.items():
                self.scaffold.morphologies.save(name, morpho, overwrite=True)
        self.scaffold._comm.barrier()

    @classmethod
    def _swc_url(cls, archive, name):
        return f"{cls._url}{cls._files}{urllib.parse.quote(archive.lower())}/CNG%20version/{name}.CNG.swc"

    @classmethod
    def _scrape_nm(cls, names):
        # Weak DH key on neuromorpho.org
        # https://stackoverflow.com/questions/38015537/python-requests-exceptions-sslerror-dh-key-too-small
        requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ":HIGH:!DH:!aNULL"
        try:
            requests.packages.urllib3.contrib.pyopenssl.util.ssl_.DEFAULT_CIPHERS += (
                ":HIGH:!DH:!aNULL"
            )
        except AttributeError:
            # no pyopenssl support used / needed / available
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ThreadPoolExecutor() as executor:
                # Certificate issues with neuromorpho --> verify=False
                res = requests.get(cls._url + cls._meta + ",".join(names), verify=False)
                if res.status_code == 404:
                    raise SelectorError(f"'{names[0]}' is not a valid NeuroMorpho name.")
                elif res.status_code != 200:
                    raise SelectorError("NeuroMorpho API error: " + res.message)
                metas = {n: None for n in names}
                for meta in res.json()["_embedded"]["neuronResources"]:
                    del meta["_links"]
                    metas[meta["neuron_name"]] = meta
                missing = [name for name, meta in metas.items() if meta is None]
                if missing:
                    raise SelectorError(
                        ", ".join(f"'{n}'" for n in missing)
                        + " are not valid NeuroMorpho names."
                    )
                swc_urls = {n: cls._swc_url(metas[n]["archive"], n) for n in names}
                req = lambda n: requests.get(swc_urls[n], verify=False)
                sub = lambda n: (executor.submit(req, n), n)
                futures = dict(map(sub, names))
                morphos = {n: None for n in names}
                with tempfile.TemporaryDirectory() as tempdir:
                    for future in concurrent.futures.as_completed(futures.keys()):
                        name = futures[future]
                        path = tempdir + f"/{name}.swc"
                        with open(path, "w") as f:
                            f.write(future.result().text)
                        morphos[name] = Morphology.from_swc(path, meta=metas[name])
                missing = [name for name, m in morphos.items() if m is None]
                if missing:  # pragma: nocover
                    raise SelectorError(
                        "Downloading NeuroMorpho failed for "
                        + ", ".join(f"'{n}'" for n in missing)
                        + "."
                    )
                return morphos


__all__ = ["MorphologySelector", "NameSelector", "NeuroMorphoSelector"]
