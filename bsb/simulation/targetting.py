import random, numpy as np
from .. import config
from ..config import types
from ..exceptions import *
from itertools import chain


@config.dynamic(attr_name="type", auto_classmap=True)
class NeuronTargetting:
    def __boot__(self):
        self.device = self._config_parent
        self.simulation = self.device.simulation if self.device is not None else None

    def get_targets(self):
        raise NotImplementedError(
            "Targetting mechanism '{}' did not implement a `get_targets` method".format(
                self.type
            )
        )


@config.node
class CellTypeTargetting(NeuronTargetting, classmap_entry="cell_type"):
    """
    Targetting mechanism (use ``"type": "cell_type"``) to target all identifiers of
    certain cell types.
    """

    cell_types = config.attr(type=types.list(type=str), required=True)

    def get_targets(self):
        sets = [self.scaffold.get_placement_set(t) for t in self.cell_types]
        ids = []
        for set in sets:
            ids.extend(set.identifiers)
        return ids


@config.node
class RepresentativesTargetting(NeuronTargetting, classmap_entry="representatives"):
    """
    Targetting mechanism (use ``"type": "representatives"``) to target all identifiers
    of certain cell types.
    """

    cell_types = config.attr(type=types.list(type=str))

    def get_targets(self):
        filter_types = self.cell_types or self.adapter.cell_models.keys()
        target_ids = [
            cell_model.cell_type.get_placement_set().identifiers
            for cell_model in self.adapter.cell_models.values()
            if not cell_model.cell_type.relay and cell_model.name in filter_types
        ]
        if hasattr(self, "cell_types"):
            target_types = [t for t in target_types if t.name in self.cell_types]
        target_ids = [t.get_placement_set().identifiers for t in target_types]
        representatives = [
            random.choice(type_ids) for type_ids in target_ids if len(target_ids) > 0
        ]
        return representatives


@config.node
class ByIdTargetting(NeuronTargetting, classmap_entry="by_id"):
    """
    Targetting mechanism (use ``"type": "by_id"``) to target all given identifiers.
    """

    targets = config.attr(type=types.list(type=int), required=True)

    def get_targets(self):
        return self.targets


@config.node
class CylindricalTargetting(NeuronTargetting, classmap_entry="cylinder"):
    """
    Targetting mechanism (use ``"type": "cylinder"``) to target all cells in a
    horizontal cylinder (xz circle expanded along y).
    """

    cell_types = config.attr(type=types.list(type=str))
    origin = config.attr(type=types.list(type=float, size=2))
    radius = config.attr(type=float, required=True)

    def boot(self):
        if self.cell_types is None:
            self.cell_types = [m.cell_type for m in self.adapter.cell_models.values()]
        if self.origin is None:
            network = self.scaffold.configuration.network
            self.origin = [network.x / 2, network.z / 2]

    def get_targets(self):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        sets = [self.scaffold.get_placement_set(t) for t in self.cell_types]
        targets = []
        for set in sets:
            if not set.positions:
                continue
            distances = np.sum((set.positions[:, [0, 2]] - self.origin) ** 2)
            targets.extend(set.identifiers[distances <= self.radius])
        return np.array(targets)


@config.node
class SphericalTargetting(NeuronTargetting, classmap_entry="sphere"):
    """
    Targetting mechanism (use ``"type": "sphere"``) to target all cells in a sphere.
    """

    cell_types = config.attr(type=types.list(type=str))
    origin = config.attr(type=types.list(type=float, size=3), required=True)
    radius = config.attr(type=float, required=True)

    def boot(self):
        if self.cell_types is None:
            self.cell_types = [m.cell_type for m in self.adapter.cell_models.values()]

    def get_targets(self):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        sets = [self.scaffold.get_placement_set(t) for t in self.cell_types]
        targets = []
        for set in sets:
            if not set.positions:
                continue
            distances = np.sum((set.positions - self.origin) ** 2)
            targets.extend(set.identifiers[distances <= self.radius])
        return np.array(targets)


class TargetsSections:
    def target_section(self, cell):
        if not hasattr(self, "section_targetting"):
            self.section_targetting = "default"
        method_name = "_section_target_" + self.section_targetting
        if not hasattr(self, method_name):
            raise Exception(
                "Unknown section targetting type '{}'".format(self.section_targetting)
            )
        return getattr(self, method_name)(cell)

    def _section_target_default(self, cell):
        if not hasattr(self, "section_count"):
            self.section_count = "all"
        elif self.section_count != "all":
            self.section_count = int(self.section_count)
        sections = cell.sections
        if hasattr(self, "section_types"):
            ts = self.section_types
            sections = [s for s in sections if any(t in s.labels for t in ts)]
        if hasattr(self, "section_type"):
            raise ConfigurationError(
                "`section_type` is deprecated, use `section_types` instead."
            )
        if self.section_count == "all":
            return sections
        return [random.choice(sections) for _ in range(self.section_count)]
