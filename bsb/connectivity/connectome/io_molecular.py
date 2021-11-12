import numpy as np
from ..strategy import ConnectionStrategy


class ConnectomeIOMolecular(ConnectionStrategy):
    """
    Legacy implementation for the connection between inferior olive and Molecular layer interneurons.
    As this is a spillover-mediated non-synaptic connection depending on the IO to Purkinje cells, each interneuron connected
    to a PC which is receving input from one IO, is also receiving input from that IO
    """

    defaults = {"common_type": "purkinje_cell"}
    casts = {"common_type": str}

    def validate(self):
        pass

    def connect(self):
        # Gather connection information
        common_type = self.scaffold.configuration.cell_types[self.common_type]
        io_cell_type = self.from_cell_types[0]
        molecular_type = self.to_cell_types[0]
        io_cells = self.scaffold.get_cells_by_type(io_cell_type.name)

        # Get connection between molecular layer cells and Purkinje cells.
        mli_common_query = self.scaffold.query_connection_cache(
            pre=molecular_type, post=common_type
        )
        if len(mli_common_query) != 1:
            raise NotImplementedError(
                f"{type(self).__name__} expects exactly 1 connection type"
                + f" between `{molecular_type.name}` and `{common_type.name}`"
            )
        mli_to_common, cache = next(iter(mli_common_query.items()))
        if len(cache) != 1:
            raise NotImplementedError(
                f"{mli_to_common.name} created {len(cache)} different sets of"
                + f" connections, while {type(self).__name__} can only handle one."
            )
        mli_to_common_matrix = cache[0]

        io_to_common = self.scaffold.query_connection_cache(
            pre=io_cell_type, post=common_type
        )
        conn_type, cache = next(iter(io_to_common.items()))
        if len(cache) < 2:
            # No IO to purkinje connections found. Do nothing.
            return
        io_to_common_matrix = cache[1]

        # Make a dictionary of which Purkinje cell is contacted by which molecular cells.
        purkinje_dict = {}
        for conn in range(len(mli_to_common_matrix)):
            purkinje_id = mli_to_common_matrix[conn][1]
            if not purkinje_id in purkinje_dict:
                purkinje_dict[purkinje_id] = []
            purkinje_dict[purkinje_id].append(mli_to_common_matrix[conn][0])

        # Use the above dictionary to connect each IO cell to the molecular cells that
        # contact the Purkinje cells this IO cell contacts.
        io_molecular = []
        # Loop over all IO-Purkinje connections
        for io_conn in range(len(io_to_common_matrix)):
            io_id = io_to_common_matrix[io_conn][0]
            purkinje_id = io_to_common_matrix[io_conn][1]
            # No molecular cells contact this Purkinje cell
            if not purkinje_id in purkinje_dict:
                continue
            target_molecular_cells = purkinje_dict[purkinje_id]
            # Make a matrix that connects this IO cell to the target molecular cells
            matrix = np.column_stack(
                (
                    np.repeat(io_id, len(target_molecular_cells)),
                    target_molecular_cells,
                )
            )
            # Add the matrix to the output dataset.
            io_molecular.extend(matrix)
        # Store the connections.
        results = np.array(io_molecular or np.empty((0, 2)))
        self.scaffold.connect_cells(self, results)
