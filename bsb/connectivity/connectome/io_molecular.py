import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeIOMolecular(ConnectionStrategy):
    """
    Legacy implementation for the connection between inferior olive and Molecular layer interneurons.
    As this is a spillover-mediated non-synaptic connection depending on the IO to Purkinje cells, each interneuron connected
    to a PC which is receving input from one IO, is also receiving input from that IO
    """

    def validate(self):
        pass

    def connect(self):
        io_type = self.from_cell_types[0]
        print(io_type.name)
        mli_type = self.to_cell_types[0]
        print(mli_type.name)
        io_cells = self.scaffold.get_placement_set(io_type)
        print(len(io_cells))
        io_to_pc = self.scaffold.get_connectivity_set("io_to_purkinje")
        print(len(io_to_pc))
        mli_to_pc_name = mli_type.name.split("_")[0] + "_to_purkinje"
        mli_to_pc = self.scaffold.get_connectivity_set(mli_to_pc_name)
        print(len(mli_to_pc))
        # Create a lookup dict to get the IO id connected to a PC id
        pc_io = dict(zip(io_to_pc.to_identifiers, io_to_pc.from_identifiers))
        # Find the IO id for each MLI-PC connection
        io_ids = np.vectorize(pc_io.get)(mli_to_pc.to_identifiers)
        io_to_mli = np.column_stack((io_ids, mli_to_pc.from_identifiers))
        self.scaffold.connect_cells(self, io_to_mli)
