import numpy as np

from bsb import ConnectionStrategy, config, types


@config.node
class DistanceConnectivity(ConnectionStrategy):
    """
    Connect cells based on the distance between their respective somas.
    The algorithm will search for potential targets surrounding the presynaptic cells
    in a sphere with a given radius.
    """

    radius: float = config.attr(type=types.float(min=0), required=True)
    """Radius of the sphere surrounding the presynaptic cell to filter potential 
       postsynaptic targets"""

    def connect(self, presyn_collection, postsyn_collection):
        # For each presynaptic placement set
        for pre_ps in presyn_collection.placement:
            # Load all presynaptic positions
            presyn_positions = pre_ps.load_positions()
            # For each postsynaptic placement set
            for post_ps in postsyn_collection.placement:
                # Load all postsynaptic positions
                postsyn_positions = post_ps.load_positions()

                # For each presynaptic cell to connect
                for j, pre_position in enumerate(presyn_positions):
                    # We measure the distance of each postsyn cell with respect to the
                    # presyn cell
                    dist = np.linalg.norm(postsyn_positions - pre_position, axis=1)
                    # We keep only the ids that are within the sphere radius
                    ids_to_keep = np.where(dist <= self.radius)[0]
                    nb_connections = len(ids_to_keep)

                    # We create two connection location array and set their
                    # neuron ids.
                    pre_locs = np.full((nb_connections, 3), -1, dtype=int)
                    pre_locs[:, 0] = j
                    post_locs = np.full((nb_connections, 3), -1, dtype=int)
                    post_locs[:, 0] = ids_to_keep

                    self.connect_cells(pre_ps, post_ps, pre_locs, post_locs)
