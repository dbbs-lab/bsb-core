from .component import SimulationComponent


class SimulationCell(SimulationComponent):
    def boot(self):
        super().boot()
        try:
            self.cell_type = self.scaffold.get_cell_type(self.name)
        except TypeNotFoundError:
            raise TypeNotFoundError(
                "Cell type '{}' not found in '{}', all cell models need to have the name of a cell type.".format(
                    self.name, self.get_config_node()
                )
            )

    def is_relay(self):
        return self.cell_type.relay

    @property
    def relay(self):
        return self.is_relay()
