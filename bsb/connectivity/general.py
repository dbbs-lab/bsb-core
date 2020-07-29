import numpy as np
from .strategy import ConnectionStrategy, TouchingConvergenceDivergence


class Convergence(TouchingConvergenceDivergence):
    """
        Implementation of a general convergence connectivity between
        two populations of cells (this does not work with entities)
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        pre = self.from_cells[from_type.name]
        post = self.to_cells[to_type.name]
        convergence = self.convergence

        pre_post = np.zeros((convergence * len(post), 2))
        for i, neuron in enumerate(post):
            connected_pre = np.random.choice(pre[:, 0], convergence, replace=False)
            range_i = range(i * convergence, (i + 1) * convergence)
            pre_post[range_i, 0] = connected_pre.astype(int)
            pre_post[range_i, 1] = neuron[0]

        self.scaffold.connect_cells(self, pre_post)


class AllToAll(ConnectionStrategy):
    """
        All to all connectivity between two neural populations
    """

    def validate(self):
        pass

    def connect(self):
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        from_cells = self.from_cells[from_type.name]
        to_cells = self.to_cells[to_type.name]
        l = len(to_cells)
        connections = np.empty([len(from_cells) * l, 2])
        to_cell_ids = to_cells[:, 0]
        for i, from_cell in enumerate(from_cells[:, 0]):
            connections[range(i * l, (i + 1) * l), 0] = from_cell
            connections[range(i * l, (i + 1) * l), 1] = to_cell_ids
        self.scaffold.connect_cells(self, connections)
