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
        # Gather connection information
        io_cell_type = self.from_cell_types[0]
        molecular_cell_type = self.to_cell_types[0]
        io_cells = self.scaffold.get_cells_by_type(io_cell_type.name)

        # Get connection between molecular layer cells and Purkinje cells.
        molecular_cell_purkinje_connections = self.scaffold.get_connection_cache_by_cell_type(
            postsynaptic="purkinje_cell", presynaptic=molecular_cell_type.name
        )

        # Extract a list of cell types objects that are sources in the MLI to PC connections.
        # molecular_cell_purkinje_connections has the connection object from which we need to extract info as the first element
        sources_mli_types = molecular_cell_purkinje_connections[0][0].from_cell_types
        # Associate an index to each MLI type which is connected to Purkinje cells
        index_mli_type = next(
            (
                index
                for (index, d) in enumerate(sources_mli_types)
                if d.name == molecular_cell_type.name
            ),
            None,
        )
        # second, third etc element in molecular_cell_purkinje_connections are the connection matrix for each element in from_cell_types
        molecular_cell_purkinje_matrix = molecular_cell_purkinje_connections[0][
            index_mli_type + 1
        ]

        io_cell_purkinje_connections = self.scaffold.get_connection_cache_by_cell_type(
            postsynaptic="purkinje_cell", presynaptic=io_cell_type.name
        )
        if len(io_cell_purkinje_connections[0]) < 2:
            # No IO to purkinje connections found. Do nothing.
            return
        io_cell_purkinje_matrix = io_cell_purkinje_connections[0][1]

        # Make a dictionary of which Purkinje cell is contacted by which molecular cells.
        purkinje_dict = {}
        for conn in range(len(molecular_cell_purkinje_matrix)):
            purkinje_id = molecular_cell_purkinje_matrix[conn][1]
            if not purkinje_id in purkinje_dict:
                purkinje_dict[purkinje_id] = []
            purkinje_dict[purkinje_id].append(molecular_cell_purkinje_matrix[conn][0])

        # Use the above dictionary to connect each IO cell to the molecular cells that
        # contact the Purkinje cells this IO cell contacts.
        io_molecular = []
        # Loop over all IO-Purkinje connections
        for io_conn in range(len(io_cell_purkinje_matrix)):
            io_id = io_cell_purkinje_matrix[io_conn][0]
            purkinje_id = io_cell_purkinje_matrix[io_conn][1]
            # No molecular cells contact this Purkinje cell
            if not purkinje_id in purkinje_dict:
                continue
            target_molecular_cells = purkinje_dict[purkinje_id]
            # Make a matrix that connects this IO cell to the target molecular cells
            matrix = np.column_stack(
                (np.repeat(io_id, len(target_molecular_cells)), target_molecular_cells,)
            )
            # Add the matrix to the output dataset.
            io_molecular.extend(matrix)
        # Store the connections.
        results = np.array(io_molecular or np.empty((0, 2)))
        self.scaffold.connect_cells(self, results)
