class Statistics:
    def __init__(self, scaffoldInstance):
        self.scaffold = scaffoldInstance
        self.cells_placed = {k: 0 for k in self.scaffold.configuration.cell_types}

    @property
    def connections(self):
        return {cs.tag: len(cs) for cs in self.scaffold.get_connectivity_sets()}
