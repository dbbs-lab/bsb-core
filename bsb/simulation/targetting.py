import itertools
import random
import numpy as np
from .. import config
from ..config import refs, types


@config.dynamic(attr_name="strategy", default="all", auto_classmap=True)
class Targetting:
    type = config.attr(type=types.in_(["cell", "connection"]), default="cell")

    def get_targets(self, cells, connections):
        if self.type == "cell":
            return cells
        elif self.type == "connection":
            return connections


@config.node
class CellTargetting(Targetting, classmap_entry="all"):
    @config.property
    def type(self):
        return "cell"

    def get_targets(self, cells, connections):
        return cells


@config.node
class ConnectionTargetting(Targetting, classmap_entry="all_connections"):
    @config.property
    def type(self):
        return "connection"

    def get_targets(self, cells, connections):
        return connections


@config.node
class CellModelTargetting(CellTargetting, classmap_entry="cell_model"):
    """
    Targetting mechanism (use ``"type": "cell_model"``) to target all cells of
    certain cell models.
    """

    cell_models = config.reflist(refs.sim_cell_model_ref, required=True)

    def get_targets(self, cells, connections):
        return [cell for cell in cells.values() if cell.model in self.cell_models]


@config.node
class RepresentativesTargetting(CellModelTargetting, classmap_entry="representatives"):
    """
    Targetting mechanism (use ``"type": "representatives"``) to target all identifiers
    of certain cell types.
    """

    n = config.attr(type=int, default=1)

    def get_targets(self, cells, connections):
        reps = {cell_model: [] for cell_model in self.cell_models}
        for cell in cells.values():
            reps[cell.model] = cell
        return [
            *itertools.chain.from_iterable(
                random.choices(group, k=self.n) for group in reps.values()
            )
        ]


@config.node
class ByIdTargetting(CellTargetting, classmap_entry="by_id"):
    """
    Targetting mechanism (use ``"type": "by_id"``) to target all given identifiers.
    """

    ids = config.attr(type=types.list(type=int), required=True)

    def get_targets(self, cells, connections):
        return [cells[id] for id in self.ids]


@config.node
class ByLabelTargetting(CellTargetting, classmap_entry="by_label"):
    """
    Targetting mechanism (use ``"type": "by_label"``) to target all given labels.
    """

    labels = config.attr(type=types.list(type=str), required=True)

    def get_targets(self, cells, connections):
        raise NotImplementedError("Labels still need to be transferred onto models")


class CellModelFilter:
    cell_models = config.reflist(refs.sim_cell_model_ref)

    def get_targets(self, cells, connections):
        return [cell for cell in cells.values() if cell.cell_model in self.cell_models]


@config.node
class CylindricalTargetting(CellModelFilter, CellTargetting, classmap_entry="cylinder"):
    """
    Targetting mechanism (use ``"type": "cylinder"``) to target all cells in a
    horizontal cylinder (xz circle expanded along y).
    """

    origin = config.attr(type=types.list(type=float, size=2))
    axis = config.attr(type=types.in_(["x", "y", "z"]), default="y")
    radius = config.attr(type=float, required=True)

    def get_targets(self, cells, connections):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        cells = super().get_targets(cells, connections)
        if self.axis == "x":
            axes = [1, 2]
        elif self.axis == "y":
            axes = [0, 2]
        else:
            axes = [0, 1]
        return [
            cell
            for cell in cells
            if np.sum((cell.position[axes] - self.origin) ** 2) < self.radius**2
        ]


@config.node
class SphericalTargetting(CellModelFilter, CellTargetting, classmap_entry="sphere"):
    """
    Targetting mechanism (use ``"type": "sphere"``) to target all cells in a sphere.
    """

    origin = config.attr(type=types.list(type=float, size=3), required=True)
    radius = config.attr(type=float, required=True)

    def get_targets(self, cells, connections):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        return [
            cell
            for cell in super().get_targets(cells, connections)
            if np.sum((cell.position - self.origin) ** 2) < self.radius**2
        ]


@config.dynamic(
    attr_name="strategy",
    default="everywhere",
    auto_classmap=True,
    classmap_entry="everywhere",
)
class LocationTargetting:
    def get_locations(self, cell):
        return cell.locations


@config.node
class SomaTargetting(LocationTargetting, classmap_entry="soma"):
    def get_locations(self, cell):
        return [cell.locations[(0, 0)]]
