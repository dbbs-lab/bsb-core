from bsb import config
from bsb.config import types, refs
from bsb.simulation.device import DeviceModel
from bsb.simulation.targetting import CellTargetting


@config.node
class NestRule:
    rule = config.attr(type=str, required=True)
    constants = config.catch_all(type=types.any_())
    cell_models = config.reflist(refs.sim_cell_model_ref)


@config.dynamic(attr_name="device", auto_classmap=True, default="external")
class NestDevice(DeviceModel):
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=float, required=True)
    targetting = config.attr(type=types.or_(CellTargetting, NestRule))


@config.node
class ExtNestDevice(NestDevice, classmap_entry="external"):
    nest_model = config.attr(type=str, required=True)
    constants = config.dict(type=types.or_(types.number(), str))

    def implement(self, adapter, simdata):
        simdata.devices[self] = device = adapter.nest.Create(
            self.nest_model, params=self.constants
        )
        if isinstance(self.targetting, CellTargetting):
            nodes = sum(
                simdata.populations[model][targets]
                for model, targets in self.targetting.get_targets(
                    adapter, simdata
                ).items()
            )
        else:
            nodes = sum(
                simdata.populations[model][targets]
                for model, targets in simdata.populations.items()
                if not self.targetting.cell_models or model in self.targetting.cell_models
            )
        adapter.nest.Connect(
            device, nodes, syn_spec={"weight": self.weight, "delay": self.delay}
        )
