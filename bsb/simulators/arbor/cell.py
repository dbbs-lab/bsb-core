import abc
import typing

from bsb import config
from bsb.config import types
from bsb.exceptions import ConfigurationError
from bsb.simulation.cell import CellModel
import arbor

from .adapter import SingleReceiverCollection

if typing.TYPE_CHECKING:
    from bsb.storage.interfaces import PlacementSet


@config.dynamic(
    attr_name="model_strategy",
    auto_classmap=True,
    required=True,
    classmap_entry=None,
)
class ArborCell(CellModel):
    gap = config.attr(type=bool, default=False)
    model = config.attr(type=types.class_(), required=True)

    @abc.abstractmethod
    def cache_population_data(self, simdata, ps: "PlacementSet"):
        pass

    @abc.abstractmethod
    def discard_population_data(self):
        pass

    @abc.abstractmethod
    def get_prefixed_catalogue(self):
        pass

    @abc.abstractmethod
    def get_cell_kind(self, gid):
        pass

    @abc.abstractmethod
    def make_receiver_collection(self):
        pass

    def get_description(self, gid):
        morphology, labels, decor = self.model.cable_cell_template()
        labels = self._add_labels(gid, labels, morphology)
        decor = self._add_decor(gid, decor)
        cc = arbor.cable_cell(morphology, labels, decor)
        return cc


@config.node
class LIFCell(ArborCell, classmap_entry="lif"):
    model = config.unset()
    constants = config.dict(type=types.any_())

    def cache_population_data(self, simdata, ps: "PlacementSet"):
        pass

    def discard_population_data(self):
        pass

    def get_prefixed_catalogue(self):
        return None, None

    def get_cell_kind(self, gid):
        return arbor.cell_kind.lif

    def get_description(self, gid):
        cell = arbor.lif_cell(f"-1_-1", f"-1_-1_0")
        try:
            for k, v in self.constants.items():
                setattr(cell, k, v)
        except AttributeError:
            node_name = type(self).constants.get_node_name(self)
            raise ConfigurationError(
                f"'{k}' is not a valid LIF parameter in '{node_name}'."
            ) from None
        return cell

    def make_receiver_collection(self):
        return SingleReceiverCollection()
