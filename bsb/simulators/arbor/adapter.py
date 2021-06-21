from ...simulation import (
    SimulatorAdapter,
    SimulationComponent,
    SimulationCell,
    TargetsNeurons,
    TargetsSections,
    SimulationResult,
    SimulationRecorder,
)
from ...reporting import report, warn
from ...exceptions import *
from ...helpers import continuity_list, continuity_hop, get_configurable_class
import numpy as np

try:
    import arbor

    _has_arbor = True
except ImportError:
    _has_arbor = False
    import types

    arbor = types.ModuleType("arbor")
    arbor.recipe = type("mock_recipe", (), dict())


class ArborCell(SimulationCell):
    node_name = "simulations.?.cell_models"

    def validate(self):
        if _has_arbor and not self.relay:
            self.model_class = get_configurable_class(self.model)

    def get_description(self):
        print("Create cable cell for", self.name)
        if not self.relay:
            return self.model_class.cable_cell()
        else:
            return arbor.spike_source_cell()


class ArborDevice(SimulationCell):
    pass


class ArborConnection(SimulationComponent):
    pass


class QuickContains:
    def __init__(self, cell_model, ps):
        self._model = cell_model
        self._ps = ps
        self._type = ps.type
        if cell_model.relay or ps.type.entity:
            print("Adding", cell_model.name, "spike source")
            self._kind = arbor.cell_kind.spike_source
        else:
            print("Adding", cell_model.name, "cable")
            self._kind = arbor.cell_kind.cable
        self._ranges = [
            (start, start + count)
            for start, count in continuity_hop(iter(ps.identifier_set.get_dataset()))
        ]

    def __contains__(self, i):
        return any(i >= start and i <= stop for start, stop in self._ranges)


class QuickLookup:
    def __init__(self, adapter):
        network = adapter.scaffold
        self._contains = [
            QuickContains(model, network.get_placement_set(model.name))
            for model in adapter.cell_models.values()
        ]

    def lookup_kind(self, gid):
        return self._lookup(gid)._kind

    def lookup_model(self, gid):
        return self._lookup(gid)._model

    def _lookup(self, gid):
        try:
            return next(c for c in self._contains if gid in c)
        except StopIteration:
            raise GidLookupError(f"Can't find gid {gid}.")


class ArborRecipe(arbor.recipe):
    def __init__(self, adapter):
        super().__init__()
        self._adapter = adapter
        self._catalogue = arbor.default_catalogue()
        self._catalogue.extend(arbor.dbbs_catalogue(), "")
        self._global_properties = arbor.neuron_cable_properties()
        self._global_properties.register(self._catalogue)
        self._lookup = QuickLookup(adapter)

    def num_cells(self):
        network = self._adapter.scaffold
        print(
            "Datasets contain",
            sum(
                len(ps) for ps in map(network.get_placement_set, network.get_cell_types())
            ),
            "cells",
        )
        s = sum(
            len(ps) for ps in map(network.get_placement_set, network.get_cell_types())
        )
        print("alive")
        return s

    def cell_kind(self, gid):
        return self._lookup.lookup_kind(gid)

    def cell_description(self, gid):
        print("Looking for models")
        model = self._lookup.lookup_model(gid)
        return model.get_description()

    def global_properties(self, kind):
        return self._global_properties


class ArborAdapter(SimulatorAdapter):
    simulator_name = "arbor"

    configuration_classes = {
        "cell_models": ArborCell,
        "connection_models": ArborConnection,
        "devices": ArborDevice,
    }

    def validate(self):
        pass

    def prepare(self):
        print("context")
        context = arbor.context()
        print("recipe")
        recipe = self.get_recipe()
        print("domains")
        domains = arbor.partition_load_balance(recipe, context)
        print("simulate")
        return arbor.simulation(recipe, domains, context)

    def simulate(self, simulation):
        print("in simulate")
        simulation.record(arbor.spike_recording.all)
        self.sampler = simulation.sample((0, 0), arbor.regular_schedule(0.1))
        simulation.run(tfinal=5000)

    def collect_output(self, simulation):
        spikes = simulation.spikes()
        data, meta = simulation.samples(handle)[0]

    def get_recipe(self):
        return ArborRecipe(self)
