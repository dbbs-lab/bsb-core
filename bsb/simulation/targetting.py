import random
import numpy as np
from numpy.random import default_rng
from .. import config
from ..config import refs, types


@config.dynamic(attr_name="strategy", default="all", auto_classmap=True)
class Targetting:
    type = config.attr(type=types.in_(["cell", "connection"]), default="cell")

    def get_targets(self, adapter, simulation, simdata):
        if self.type == "cell":
            return simdata.populations
        elif self.type == "connection":
            return simdata.connections


@config.node
class CellTargetting(Targetting, classmap_entry="all"):
    @config.property
    def type(self):
        return "cell"

    def get_targets(self, adapter, simulation, simdata):
        return simdata.populations


@config.node
class ConnectionTargetting(Targetting, classmap_entry="all_connections"):
    @config.property
    def type(self):
        return "connection"

    def get_targets(self, adapter, simulation, simdata):
        return simdata.connections


class CellModelFilter:
    cell_models = config.reflist(refs.sim_cell_model_ref, required=True)

    def get_targets(self, adapter, simulation, simdata):
        return {
            model: data
            for model, data in simdata.placement.items()
            if model in self.cell_models
        }


@config.node
class CellModelTargetting(CellModelFilter, CellTargetting, classmap_entry="cell_model"):
    """
    Targets all cells of certain cell models.
    """

    # Inherits from CellModelFilter mixin, to avoid having to use `compose_nodes` in other
    # child classes.
    pass


@config.node
class RepresentativesTargetting(CellModelTargetting, classmap_entry="representatives"):
    """
    Targets all identifiers of certain cell types.
    """

    n = config.attr(type=int, default=1)

    def get_targets(self, adapter, simulation, simdata):
        return {
            model: default_rng().choice(len(data), size=self.n, replace=False)
            for model, data in simdata.placement
            if model in self.cell_models
        }


@config.node
class ByIdTargetting(CellTargetting, classmap_entry="by_id"):
    """
    Targets all given identifiers.
    """

    ids = config.attr(type=types.dict(type=types.list(type=int)), required=True)

    def get_targets(self, adapter, simulation, simdata):
        by_name = {model.name: model for model in simdata.populations.keys()}
        return {
            model: ids
            for model_name, ids in self.ids.items()
            if (model := by_name.get(model_name)) is not None
        }


@config.node
class ByLabelTargetting(CellTargetting, classmap_entry="by_label"):
    """
    Targets all given labels.
    """

    labels = config.attr(type=types.list(type=str), required=True)

    def get_targets(self, adapter, simulation, simdata):
        raise NotImplementedError("Labels still need to be transferred onto models")


@config.node
class CylindricalTargetting(CellModelFilter, CellTargetting, classmap_entry="cylinder"):
    """
    Targets all cells in a cylinder along specified axis.
    """

    origin = config.attr(type=types.list(type=float, size=2))
    axis = config.attr(type=types.in_(["x", "y", "z"]), default="y")
    radius = config.attr(type=float, required=True)

    def get_targets(self, adapter, simulation, simdata):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        if self.axis == "x":
            axes = [1, 2]
        elif self.axis == "y":
            axes = [0, 2]
        else:
            axes = [0, 1]
        return {
            model: np.nonzero(
                np.sum(data.load_positions()[axes] - self.origin**2, axis=0)
                < self.radius**2
            )[0]
            for model, data in simdata.placement.items()
            if model in self.cell_models
        }


@config.node
class SphericalTargetting(CellModelFilter, CellTargetting, classmap_entry="sphere"):
    """
    Targets all cells in a sphere.
    """

    origin = config.attr(type=types.list(type=float, size=3), required=True)
    radius = config.attr(type=float, required=True)

    def get_targets(self, adapter, simulation, simdata):
        """
        Target all or certain cells within a cylinder of specified radius.
        """
        return {
            model: np.nonzero(
                np.sum(data.load_positions() - self.origin**2, axis=0)
                < self.radius**2
            )[0]
            for model, data in simdata.placement.items()
            if model in self.cell_models
        }


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
