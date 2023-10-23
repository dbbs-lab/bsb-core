import types
import numpy as np

from ..config._attrs import cfgdict, cfglist
from time import time
import itertools
from .. import config
from ._backends import get_simulation_nodes
from .cell import CellModel
from .connection import ConnectionModel
from .device import DeviceModel
from ..config import types as cfgtypes
import typing

if typing.TYPE_CHECKING:
    from ..core import Scaffold
    from ..connectivity import ConnectionStrategy
    from ..cell_types import CellType
    from ..storage.interfaces import ConnectivitySet


class ProgressEvent:
    def __init__(self, progression, duration, time):
        self.progression = progression
        self.duration = duration
        self.time = time


@config.pluggable(key="simulator", plugin_name="simulation backend")
class Simulation:
    scaffold: "Scaffold"
    name: str = config.attr(key=True)
    duration: float = config.attr(type=float, required=True)
    cell_models: cfgdict[CellModel] = config.slot(type=CellModel, required=True)
    connection_models: cfgdict[ConnectionModel] = config.slot(
        type=ConnectionModel, required=True
    )
    devices: cfgdict[DeviceModel] = config.slot(type=DeviceModel, required=True)
    post_prepare: cfglist[
        typing.Callable[["Simulation", typing.Any], None]
    ] = config.list(type=cfgtypes.function_())

    @staticmethod
    def __plugins__():
        return get_simulation_nodes()

    def start_progress(self, duration):
        """
        Start a progress meter.
        """
        self._progdur = duration
        self._progstart = self._last_progtic = time()
        self._progtics = 0

    def progress(self, step):
        """
        Report simulation progress.
        """
        now = time()
        tic = now - self._last_progtic
        self._progtics += 1
        el = now - self._progstart
        # report(
        #     f"Simulated {step}/{self._progdur}ms.",
        #     f"{el:.2f}s elapsed.",
        #     f"Simulated tick in {tic:.2f}.",
        #     f"Avg tick {el / self._progtics:.4f}s",
        #     level=3,
        #     ongoing=False,
        # )
        progress = types.SimpleNamespace(
            progression=step, duration=self._progdur, time=time()
        )
        self._last_progtic = now
        return progress

    def step_progress(self, duration, step=1):
        steps = itertools.chain(np.arange(0, duration), (duration,))
        a, b = itertools.tee(steps)
        next(b, None)
        yield from zip(a, b)

    def add_progress_listener(self, listener):
        self._progress_listeners.append(listener)

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
