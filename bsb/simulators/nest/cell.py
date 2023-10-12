from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel


@config.node
class NestCell(CellModel):
    model = config.attr(type=str, default="iaf_psc_alpha")
    constants = config.dict(type=types.any_())

    def create_population(self, simdata):
        import nest

        n = len(simdata.placement[self])
        population = nest.Create(self.model, n) if n else nest.NodeCollection([])
        self.set_constants(population)
        self.set_parameters(population, simdata)
        return population

    def set_constants(self, population):
        population.set(self.constants)

    def set_parameters(self, population, simdata):
        ps = simdata.placement[self]
        for param in self.parameters:
            population.set(param.name, param.get_value(ps))
