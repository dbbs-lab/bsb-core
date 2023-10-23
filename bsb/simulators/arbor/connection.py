import tqdm

from bsb import config
from bsb.simulation.connection import ConnectionModel
import arbor


class Receiver:
    def __init__(self, conn_model, from_gid, loc_from, loc_on, index=-1):
        self.conn_model = conn_model
        self.from_gid = from_gid
        self.loc_from = loc_from
        self.loc_on = loc_on
        self.synapse = arbor.synapse("expsyn")
        self.index = index

    def from_(self):
        b, p = self.loc_from
        return arbor.cell_global_label(self.from_gid, f"{b}_{p}")

    def on(self):
        # self.index is set on us by the ReceiverCollection when we are appended.
        b, p = self.loc_on
        return arbor.cell_local_label(f"{b}_{p}_{self.index}")

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


@config.node
class ArborConnection(ConnectionModel):
    gap = config.attr(type=bool, default=False)
    weight = config.attr(type=float, required=True)
    delay = config.attr(type=float, required=True)

    def create_gap_junctions_on(self, gj_on_gid, conns):
        for pre_loc, post_loc in conns:
            conn = Connection(pre_loc, post_loc)
            gj_on_gid.setdefault(conn.from_id, []).append(conn)

    def create_connections_on(self, conns_on_gid, conns, pop_pre, pop_post):
        for pre_loc, post_loc in tqdm.tqdm(conns, total=len(conns), desc=self.name):
            conns_on_gid[post_loc[0] + pop_post.offset].append(
                Receiver(self, pre_loc[0] + pop_pre.offset, pre_loc[1:], post_loc[1:])
            )

    def create_connections_from(self, conns_from_gid, conns, pop_pre, pop_post):
        for pre_loc, post_loc in conns:
            conns_from_gid[int(pre_loc[0] + pop_pre.offset)].append(pre_loc[1:])

    def gap_junction(self, conn):
        l = arbor.cell_local_label(f"gap_{conn.to_compartment.id}")
        g = arbor.cell_global_label(int(conn.from_id), f"gap_{conn.from_compartment.id}")
        return arbor.gap_junction_connection(g, l, self.weight)
