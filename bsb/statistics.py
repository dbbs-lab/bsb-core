class Statistics:
    def __init__(self, scaffold):
        self.scaffold = scaffold
        self.cells_placed = CellsPlaced(scaffold)

    @property
    def connections(self):
        return {cs.tag: len(cs) for cs in self.scaffold.get_connectivity_sets()}


class CellsPlaced:
    def __init__(self, scaffold):
        self.func = lambda name: len(scaffold.get_placement_set(name))

    def __getattr__(self, name):
        return self.func(name)

    def __getitem__(self, name):
        return self.func(name)
