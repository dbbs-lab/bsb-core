import typing
from time import time

from .. import config
from ..config import types as cfgtypes
from ..config._attrs import cfgdict, cfglist
from ._backends import get_simulation_nodes
from .cell import CellModel
from .connection import ConnectionModel
from .device import DeviceModel

if typing.TYPE_CHECKING:
    from ..cell_types import CellType
    from ..connectivity import ConnectionStrategy
    from ..core import Scaffold
    from ..storage.interfaces import ConnectivitySet


class ProgressEvent:
    def __init__(self, progression, duration, time):
        self.progression = progression
        self.duration = duration
        self.time = time


@config.pluggable(key="simulator", plugin_name="simulation backend")
class Simulation:
    scaffold: "Scaffold"
    simulator: str
    name: str = config.attr(key=True)
    duration: float = config.attr(type=float, required=True)
    cell_models: cfgdict[CellModel] = config.slot(type=CellModel, required=True)
    connection_models: cfgdict[ConnectionModel] = config.slot(
        type=ConnectionModel, required=True
    )
    devices: cfgdict[DeviceModel] = config.slot(type=DeviceModel, required=True)
    post_prepare: cfglist[typing.Callable[["Simulation", typing.Any], None]] = (
        config.list(type=cfgtypes.function_())
    )

    @staticmethod
    def __plugins__():
        return get_simulation_nodes()

    def get_model_of(
        self, type: typing.Union["CellType", "ConnectionStrategy"]
    ) -> typing.Optional[typing.Union["CellModel", "ConnectionModel"]]:
        cell_models = [cm for cm in self.cell_models.values() if cm.cell_type is type]
        if cell_models:
            return cell_models[0]
        conn_models = [
            cm for cm in self.connection_models.values() if cm.connection_type is type
        ]
        if conn_models:
            return conn_models[0]

    def get_connectivity_sets(
        self,
    ) -> typing.Mapping["ConnectionModel", "ConnectivitySet"]:
        return {
            model: self.scaffold.get_connectivity_set(model.name)
            for model in sorted(self.connection_models.values())
        }


__all__ = ["ProgressEvent", "Simulation"]
