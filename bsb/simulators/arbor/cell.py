import abc
import typing

from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel
from bsb.exceptions import AdapterError
from bsb.reporting import warn
import itertools as _it
import arbor

if typing.TYPE_CHECKING:
    from bsb.storage.interfaces import PlacementSet


@config.dynamic(
    attr_name="model_strategy",
    auto_classmap=True,
    required=True,
    classmap_entry=None,
)
class ArborCell(CellModel, abc.ABC):
    gap = config.attr(type=bool, default=False)
    model = config.attr(type=types.class_(), required=True)

    @abc.abstractmethod
    def cache_population_data(self, simdata, ps: "PlacementSet"):
        pass

    @abc.abstractmethod
    def discard_population_data(self):
        pass

    @abc.abstractmethod
    def get_prefixed_catalogue(self):
        pass

    def get_description(self, gid):
        morphology, labels, decor = self.model.cable_cell_template()
        labels = self._add_labels(gid, labels, morphology)
        decor = self._add_decor(gid, decor)
        cc = arbor.cable_cell(morphology, labels, decor)
        return cc

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


@config.node
class LIFCell(ArborCell, classmap_entry="lif"):
    model = config.unset()
    constants = config.dict(type=types.any_())

    def cache_population_data(self, simdata, ps: "PlacementSet"):
        pass

    def discard_population_data(self):
        pass

    def get_prefixed_catalogue(self):
        return None, None

    def get_description(self, gid):
        cell = arbor.lif_cell(f"ello{gid}", f"bye{gid}")
        for k, v in self.constants:
            setattr(cell, k, v)
        return cell
