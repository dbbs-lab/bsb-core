from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel
from bsb.exceptions import AdapterError
from bsb.reporting import warn
import itertools as _it
import collections

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


def _consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(_it.islice(iterator, n, n), None)


_it.consume = _consume


@config.node
class ArborCell(CellModel):
    node_name = "simulations.?.cell_models"
    model = config.attr(
        type=types.class_, required=lambda s: "relay" not in s or not s["relay"]
    )
    default_endpoint = "comp_-1"

    def get_description(self, gid):
        if not self.relay:
            morphology, labels, decor = self.model.cable_cell_template()
            labels = self._add_labels(gid, labels, morphology)
            decor = self._add_decor(gid, decor)
            cc = arbor.cable_cell(morphology, labels, decor)
            return cc
        else:
            schedule = self.get_schedule(gid)
            return arbor.spike_source_cell(self.default_endpoint, schedule)

    def get_schedule(self, gid):
        schedule = arbor.explicit_schedule([])
        for device in self.adapter._devices_on[gid]:
            pattern = device.get_pattern(gid)
            if not pattern:
                continue
            merged = pattern + schedule.events(0, float("inf"))
            schedule = arbor.explicit_schedule(merged)
        return schedule

    def _add_decor(self, gid, decor):
        self._soma_detector(decor)
        self._create_transmitters(gid, decor)
        self._create_gaps(gid, decor)
        self._create_receivers(gid, decor)
        return decor

    def _add_labels(self, gid, labels, morphology):
        pwlin = arbor.place_pwlin(morphology)

        def comp_label(comp):
            if comp.id == -1:
                warn(f"Encountered nil compartment on {gid}")
                return
            loc, d = pwlin.closest(*comp.start)
            if d > 0.0001:
                raise AdapterError(f"Couldn't find {comp.start}, on {self._str(gid)}")
            labels[f"comp_{comp.id}"] = str(loc)

        comps_from = self.adapter._connections_from[gid]
        comps_on = (rcv.comp_on for rcv in self.adapter._connections_on[gid])
        gaps = (c.to_compartment for c in self.adapter._gap_junctions_on.get(gid, []))
        _it.consume(comp_label(i) for i in _it.chain(comps_from, comps_on, gaps))
        labels[self.default_endpoint] = "(root)"
        return labels

    def _str(self, gid):
        return f"{self.adapter._name_of(gid)} {gid}"

    def _soma_detector(self, decor):
        decor.place("(root)", arbor.spike_detector(-10), self.default_endpoint)

    def _create_transmitters(self, gid, decor):
        done = set()
        for comp in self.adapter._connections_from[gid]:
            if comp.id in done:
                continue
            else:
                done.add(comp.id)
            decor.place(f'"comp_{comp.id}"', arbor.spike_detector(-10), f"comp_{comp.id}")

    def _create_gaps(self, gid, decor):
        done = set()
        for conn in self.adapter._gap_junctions_on.get(gid, []):
            comp = conn.to_compartment
            if comp.id in done:
                continue
            else:
                done.add(comp.id)
            decor.place(f'"comp_{comp.id}"', arbor.junction("gj"), f"gap_{comp.id}")

    def _create_receivers(self, gid, decor):
        for rcv in self.adapter._connections_on[gid]:
            decor.place(
                f'"comp_{rcv.comp_on.id}"',
                rcv.synapse,
                f"comp_{rcv.comp_on.id}_{rcv.index}",
            )
