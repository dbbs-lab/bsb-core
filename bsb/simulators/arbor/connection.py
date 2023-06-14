import tqdm

from bsb import config
from bsb.simulation.connection import ConnectionModel
import arbor


class Receiver:
    def __init__(self, conn_model, from_gid, loc_from, loc_on):
        self.conn_model = conn_model
        self.from_gid = from_gid
        self.loc_from = loc_from
        self.loc_on = loc_on
        self.synapse = arbor.synapse("expsyn")

    def from_(self):
        return arbor.cell_global_label(self.from_gid, f"comp_{self.loc_from}")

    def on(self):
        # Not sure if endpoint labels need to be unique anymore, what about LIF with only
        # 1 source and target label?

        # # self.index is set on us by the ReceiverCollection when we are appended.
        # return arbor.cell_local_label(f"comp_{self.loc_on}_{self.index}")
        return arbor.cell_local_label(f"comp_{self.loc_on}")

    @property
    def weight(self):
        return self.conn_model.weight

    @property
    def delay(self):
        return self.conn_model.delay


class Connection:
    def __init__(self, pre_loc, post_loc):
        self.from_id = pre_loc[0]
        self.to_id = post_loc[0]
        self.pre_loc = pre_loc[1:]
        self.post_loc = post_loc[1:]


class ArborConnection(ConnectionModel):
    defaults = {"gap": False, "delay": 0.025, "weight": 1.0}
    casts = {"delay": float, "gap": bool, "weight": float}

    def validate(self):
        pass

    def make_receiver(*args):
        return Receiver(*args)

    def gap_(self, conn):
        l = arbor.cell_local_label(f"gap_{conn.to_compartment.id}")
        g = arbor.cell_global_label(int(conn.from_id), f"gap_{conn.from_compartment.id}")
        return arbor.gap_junction_connection(g, l, self.weight)
