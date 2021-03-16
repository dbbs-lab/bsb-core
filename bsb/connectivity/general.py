import os, numpy as np
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

class ExternalConnections(ConnectionStrategy):
    """
    Load the connection matrix from an external source.
    """
    required = ["source"]
    casts = {"format": str, "warn_missing": bool, "use_map": bool, "headers": bool}
    defaults = {
        "format": "csv",
        "headers": True,
        "use_map": False,
        "warn_missing": True,
        "delimiter": ",",
    }

    has_external_source = True

    def check_external_source(self):
        return os.path.exists(self.source)

    def get_external_source(self):
        return self.source

    def validate(self):
        if self.warn_missing and not self.check_external_source():
            src = self.get_external_source()
            warn(f"Missing external source '{src}' for '{self.name}'")

    def connect(self):
        if self.format == "csv":
            return self._connect_from_csv()

    def _connect_from_csv(self):
        if not self.check_external_source():
            src = self.get_external_source()
            raise RuntimeError(f"Missing source file '{src}' for `{self.name}`.")
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        # Read the entire csv, skipping the headers if there are any.
        data = np.loadtxt(
            self.get_external_source(),
            skiprows=int(self.headers),
            delimiter=self.delimiter,
        )
        if self.use_map:
            from_gid_map = self.scaffold.load_appendix(from_type.name + "_ext_map")
            to_gid_map = self.scaffold.load_appendix(to_type.name + "_ext_map")
            data[0, :] = np.vectorize(from_gid_map.get)(data[0, :])
            data[1, :] = np.vectorize(to_gid_map.get)(data[1, :])
        self.scaffold.connect_cells(self, data)
