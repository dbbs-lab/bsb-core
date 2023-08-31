import warnings

from bsb import config
from bsb.config import types, refs
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import Targetting


@config.node
class NestRule:
    rule = config.attr(type=str, required=True)
    constants = config.catch_all(type=types.any_())
    cell_models = config.reflist(refs.sim_cell_model_ref)


@config.dynamic(attr_name="device", auto_classmap=True, default="external")
class NestDevice(DeviceModel):
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=float, required=True)
    targetting = config.attr(
        type=types.or_(Targetting, NestRule), default=dict, call_default=True
    )

    def get_target_nodes(self, adapter, simulation, simdata):
        if isinstance(self.targetting, Targetting):
            node_collector = self.targetting.get_targets(
                adapter, simulation, simdata
            ).values()
        else:
            node_collector = (
                simdata.populations[model][targets]
                for model, targets in simdata.populations.items()
                if not self.targetting.cell_models or model in self.targetting.cell_models
            )
        return sum(node_collector, start=adapter.nest.NodeCollection())

    def connect_to_nodes(self, device, nodes):
        import nest

        if len(nodes) == 0:
            warnings.warn(f"{self.name} has no targets")
        else:
            try:
                nest.Connect(
                    device,
                    nodes,
                    syn_spec={"weight": self.weight, "delay": self.delay},
                )
            except Exception as e:
                if "does not send output" not in str(e):
                    raise
                nest.Connect(
                    nodes,
                    device,
                    syn_spec={"weight": self.weight, "delay": self.delay},
                )

    def register_device(self, simdata, device):
        simdata.devices[self] = device
        return device


@config.node
class ExtNestDevice(NestDevice, classmap_entry="external"):
    nest_model = config.attr(type=str, required=True)
    constants = config.dict(type=types.or_(types.number(), str))

    def implement(self, adapter, simulation, simdata):
        simdata.devices[self] = device = adapter.nest.Create(
            self.nest_model, params=self.constants
        )
        nodes = self.get_target_nodes(adapter, simdata)
        self.connect_to_nodes(device, nodes)
