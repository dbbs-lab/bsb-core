from bsb import config
from bsb.exceptions import DeviceConnectionError
from bsb.config import types
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import CellTargetting


class TargetLocation:
    def __init__(self, cell, section, connection=None):
        self.cell = cell
        self.section = section
        self.connection = connection

    def get_synapses(self):
        return self.connection and self.connection.synapses


@config.dynamic(attr_name="device", type=types.in_classmap(), auto_classmap=True)
class NeuronDevice(DeviceModel):
    radius = config.attr(type=float)
    origin = config.attr(type=types.list(type=float, size=3))
    targetting = config.attr(type=CellTargetting, required=True)
    io = config.attr(type=types.in_(["input", "output"]), required=True)

    def create_patterns(self):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `create_patterns` function."
        )

    def get_pattern(self, target, cell=None, section=None, synapse=None):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `get_pattern` function."
        )

    def implement(self, target, location):
        raise NotImplementedError(
            "The "
            + self.__class__.__name__
            + " device does not implement any `implement` function."
        )

    def get_locations(self, target):
        locations = []
        if target in self.adapter.relay_scheme:
            for cell_id, section_id, connection in self.adapter.relay_scheme[target]:
                if cell_id not in self.adapter.node_cells:
                    continue
                cell = self.adapter.cells[cell_id]
                section = cell.sections[section_id]
                locations.append(TargetLocation(cell, section, connection))
        elif target in self.adapter.node_cells:
            try:
                cell = self.adapter.cells[target]
            except KeyError:
                raise DeviceConnectionError(
                    "Missing cell {} on node {} while trying to implement device '{}'. This can occur if the cell was placed in the network but not represented with a model in the simulation config.".format(
                        target, self.adapter.get_rank(), self.name
                    )
                )
            sections = self.target_section(cell)
            locations.extend(TargetLocation(cell, section) for section in sections)
        return locations
