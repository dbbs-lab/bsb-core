from ..exceptions import *
from .. import config
from ..config import refs, types
import concurrent
from concurrent.futures import ThreadPoolExecutor
import requests
import abc
import warnings
import re
import urllib
import tempfile
from . import Morphology


@config.dynamic(
    attr_name="select",
    auto_classmap=True,
    required=False,
    default="by_name",
)
class MorphologySelector(abc.ABC):
    @abc.abstractmethod
    def validate(self, all_morphos):
        pass

    @abc.abstractmethod
    def pick(self, morphology):
        pass


@config.node
class NameSelector(MorphologySelector, classmap_entry="by_name"):
    names = config.list(type=str, required=True)

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
    _url = "https://neuromorpho.org/"
    _name = "neuron_info.jsp?neuron_name="
    _files = "dableFiles/"
    _pat = re.compile(
        r"<a href=dableFiles/(.*)>Morphology File \(Standardized\)</a>", re.MULTILINE
    )

    def __boot__(self):
        if self.scaffold.is_mpi_master:
            try:
                morphos = self._scrape_nm(self.names)
            except:
                if hasattr(self.scaffold, "MPI"):
                    self.scaffold.MPI.COMM_WORLD.Barrier()
                raise
            for name, morpho in morphos.items():
                self.scaffold.morphologies.save(name, morpho, overwrite=True)
        if hasattr(self.scaffold, "MPI"):
            self.scaffold.MPI.COMM_WORLD.Barrier()

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
                req = lambda n: requests.get(cls._url + cls._name + n, verify=False)
                sub = lambda n: (executor.submit(req, n), n)
                futures = dict(map(sub, names))
                filenames = {}
                for future in concurrent.futures.as_completed(futures.keys()):
                    name = futures[future]
                    data = future.result()
                    try:
                        file = cls._pat.search(data.text)[1]
                    except:
                        filenames[name] = None
                    else:
                        filenames[name] = file
                missing = [name for name, file in filenames.items() if file is None]
                if missing:
                    raise SelectorError(
                        ", ".join(f"'{n}'" for n in missing)
                        + " are not valid NeuroMorpho names."
                    )
                req = lambda n: requests.get(
                    cls._url + cls._files + filenames[n], verify=False
                )
                futures = dict(map(sub, names))
                morphos = {n: None for n in names}
                with tempfile.TemporaryDirectory() as tempdir:
                    for future in concurrent.futures.as_completed(futures.keys()):
                        name = futures[future]
                        data = future.result()
                        fname = urllib.parse.unquote(filenames[name]).split("/")[-1]
                        path = tempdir + f"/{fname}"
                        with open(path, "w") as f:
                            f.write(data.text)
                        try:
                            morphos[name] = Morphology.from_swc(path)
                        except:
                            morphos[name] = Morphology.from_file(path)
                missing = [name for name, m in morphos.items() if m is None]
                if missing:  # pragma: nocover
                    raise SelectorError(
                        "Downloading NeuroMorpho failed for "
                        + ", ".join(f"'{n}'" for n in missing)
                        + "."
                    )
                return morphos
