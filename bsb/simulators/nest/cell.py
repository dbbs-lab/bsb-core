from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel


@config.node
class NestCell(CellModel):
    neuron_model = config.attr(type=str)
    constants = config.dict(type=types.any_())

    def create_population(self):
        import nest

        n = len(self.get_placement_set())
        population = nest.Create(self.neuron_model, n) if n else nest.NodeCollection([])
        self.set_constants(population)
        self.set_parameters(population)
        return population

    def set_constants(self, population):
        population.set(self.constants)

    def set_parameters(self, population):
        ps = self.get_placement_set()
        for param in self.parameters:
            population.set(param.name, param.get_value(ps))
