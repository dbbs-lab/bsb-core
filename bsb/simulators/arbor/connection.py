from bsb import config
from bsb.config import types
from bsb.simulation.connection import ConnectionModel

try:
    import arbor

    _has_arbor = True
except ImportError:
    _has_arbor = False
    import types as _t

    # Mock missing requirements, as arbor is, like
    # all simulators, an optional dep. of the BSB.
    arbor = _t.ModuleType("arbor")
    arbor.recipe = type("mock_recipe", (), dict())

    def get(*arg):
        raise ImportError("Arbor not installed.")

    arbor.__getattr__ = get


class Receiver:
    def __init__(self, conn_model, from_gid, comp_from, comp_on):
        self.conn_model = conn_model
        self.from_gid = from_gid
        self.comp_from = comp_from
        self.comp_on = comp_on
        self.synapse = arbor.synapse("expsyn")


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
