import numpy as np
from ..strategy import ConnectionStrategy
from ... import config


@config.node
class ConnectomeIOPurkinje(ConnectionStrategy):
    """
        Legacy implementation for the connection between inferior olive and Purkinje cells.
    """

    def validate(self):
        pass

    def connect(self):
        io_cells = self.from_cells[self.presynaptic.type.name]
        purkinje_cells = self.to_cells[self.postsynaptic.type.name]
        io_purkinje = np.empty([len(purkinje_cells), 2])
        random_io_indices = np.random.randint(len(io_cells), size=len(purkinje_cells))
        io_purkinje[:, 0] = io_cells[random_io_indices, 0]
        io_purkinje[:, 1] = purkinje_cells[:, 0]
        self.scaffold.connect_cells(self, io_purkinje)
