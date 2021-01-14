import numpy as np
from ..strategy import ConnectionStrategy


class SatelliteCommonPresynaptic(ConnectionStrategy):
    """
    Connectivity for satellite neurons (homologous to center neurons)
    """

    def validate(self):
        pass

    def connect(self):
        config = self.scaffold.configuration
        from_type = self.from_cell_types[0]
        from_cells = self.from_cells[from_type.name][:, 0]
        to_type = self.to_cell_types[0]
        to_cells = self.to_cells[to_type.name][:, 0]
        # If we do not have pre or post synaptic cells we return an empty array
        if len(to_cells) == 0 or len(from_cells) == 0:
            self.scaffold.connect_cells(self, np.empty((0, 2)))
            return

        # We have to find the cell_type of the planets
        planet_types = to_type.placement.planet_types
        if planet_types == []:  # If the satellite does not have a planet
            self.scaffold.connect_cells(self, np.empty((0, 2)))
            return
        if len(planet_types) > 1:
            raise NotImplementedError(
                "The SatelliteCommonPresynaptic strategy for {} does not handle multiple planet types".format(
                    self.name
                )
            )

        satellites = self.scaffold.get_cells_by_type(to_type.name)[:, 0]
        satellite_map = self.scaffold._planets[to_type.name].copy()
        # Get the connections already made between the "from" cells and the planet cells
        to_planet_connections = self.scaffold.get_connection_cache_by_cell_type(
            presynaptic=from_type.name, postsynaptic=planet_types
        )  # These are the connections from the "from_cells" to the "planet" cells
        if len(to_planet_connections) != 1:
            raise NotImplementedError(
                "The SatelliteCommonPresynaptic strategy for {} handles only single connection types".format(
                    self.name
                )
            )
        if len(to_planet_connections[0]) != 2:
            raise NotImplementedError(
                "The SatelliteCommonPresynaptic strategy for {} handles only single connection sets".format(
                    self.name
                )
            )

        to_satellite_connections = np.zeros(np.shape(to_planet_connections[0][1]))
        counter = 0
        # For each connection, change the post synaptic neuron (planet) substituting the relative satellite
        for connection_i in to_planet_connections[0][1]:
            if connection_i[0] in from_cells:
                target_planet = connection_i[1]
                target_satellite = satellites[np.where(satellite_map == target_planet)]
                to_satellite_connections[counter, :] = [connection_i[0], target_satellite]
                counter += 1
        # Connect "from" cells with satellites
        self.scaffold.connect_cells(self, to_satellite_connections[:counter, :])
