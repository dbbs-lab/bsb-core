import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeIOPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connection between inferior olive and Purkinje cells.
        Purkinje cells are clustered (number of clusters is the number of IO cells), and each clusters
        is innervated by 1 IO cell
    """

    def validate(self):
        pass

    def connect(self):
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        io_cells = self.from_cells[from_type.name]
        if len(io_cells) == 0:
            self.scaffold.connect_cells(self, np.empty((0, 2)))
            return
        purkinje_cells = self.to_cells[to_type.name]
        convergence = 1  # Purkinje cells should be always constrained to receive signal from only 1 Inferior Olive neuron
        io_purkinje = np.empty([len(purkinje_cells), 2])
        for i, pc in enumerate(purkinje_cells):
            np.random.shuffle(io_cells)
            io_purkinje[i, 0] = io_cells[0, 0]
            io_purkinje[i, 1] = pc[0]
        results = io_purkinje
        self.scaffold.connect_cells(self, results)
