from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel


@config.node
class NeuronCell(CellModel):
    model = config.attr(
        type=types.class_(), required=lambda s: not ("relay" in s and s["relay"])
    )
    record_soma = config.attr(default=False)
    record_spikes = config.attr(default=False)
    entity = config.attr(default=False)

    def boot(self):
        super().boot()
        self.instances = []

    def __getitem__(self, i):
        return self.instances[i]

    def get_parameters(self):
        # Get the default synapse parameters
        params = self.parameters.copy()
        return params
