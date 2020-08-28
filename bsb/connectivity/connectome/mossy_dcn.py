import numpy as np
from ..strategy import ConnectionStrategy
from ...reporting import report, warn
from ...exceptions import *


class ConnectomeMossyDCN(ConnectionStrategy):
    """
        Implementation for the connection between mossy fibers and DCN cells.
    """

    casts = {"convergence": int}
    required = ["convergence"]

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        mossy_cell_type = self.from_cell_types[0]
        dcn_cell_type = self.to_cell_types[0]
        mossy = self.scaffold.entities_by_type[mossy_cell_type.name]
        dcn_cells = self.scaffold.cells_by_type[dcn_cell_type.name]

        convergence = self.convergence
        if convergence > len(mossy):
            convergence = len(mossy)
            warn(
                "Convergence for MF-DCN saturated at MF number. Network too small",
                ConnectivityWarning,
            )

        mf_dcn = np.zeros((convergence * len(dcn_cells), 2))
        for i, dcn in enumerate(dcn_cells):
            connected_mfs = np.random.choice(mossy, convergence, replace=False)
            range_i = range(i * convergence, (i + 1) * convergence)
            mf_dcn[range_i, 0] = connected_mfs.astype(int)
            mf_dcn[range_i, 1] = dcn[0]

        self.scaffold.connect_cells(self, mf_dcn)
