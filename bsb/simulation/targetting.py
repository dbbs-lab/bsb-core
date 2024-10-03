import functools
import math
import typing

import numpy as np
from numpy.random import default_rng

from .. import config
from ..config import refs, types

if typing.TYPE_CHECKING:
    from .cell import CellModel


@config.dynamic(attr_name="strategy", default="all", auto_classmap=True)
class Targetting:
    type: typing.Union[typing.Literal["cell"], typing.Literal["connection"]] = (
        config.attr(type=types.in_(["cell", "connection"]), default="cell")
    )

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
    cell_models: list["CellModel"] = config.reflist(
        refs.sim_cell_model_ref, required=False
    )

    def get_targets(self, adapter, simulation, simdata):
        return {
            model: pop
            for model, pop in simdata.populations.items()
            if not self.cell_models or model in self.cell_models
        }


class FractionFilter:
    count = config.attr(
        type=int, required=types.mut_excl("fraction", "count", required=False)
    )
    fraction = config.attr(
        type=types.fraction(),
        required=types.mut_excl("fraction", "count", required=False),
    )

    def satisfy_fractions(self, targets):
        return {model: self._frac(data) for model, data in targets.items()}

    def _frac(self, data):
        take = None
        if self.count is not None:
            take = self.count
        if self.fraction is not None:
            take = math.floor(len(data) * self.fraction)
        if take is None:
            return data
        else:
            # Select `take` elements from data with a boolean mask (otherwise a sorted
            # integer mask would be required)
            idx = np.zeros(len(data), dtype=bool)
            idx[np.random.default_rng().integers(0, len(data), take)] = True
            return data[idx]

    @staticmethod
    def filter(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            return self.satisfy_fractions(f(self, *args, **kwargs))

        return wrapper


@config.node
class CellModelTargetting(
    CellModelFilter, FractionFilter, CellTargetting, classmap_entry="cell_model"
):
    """
    Targets all cells of certain cell models.
    """

    cell_models: list["CellModel"] = config.reflist(
        refs.sim_cell_model_ref, required=True
    )

    @FractionFilter.filter
    def get_targets(self, adapter, simulation, simdata):
        return super().get_targets(adapter, simulation, simdata)


@config.node
class RepresentativesTargetting(
    CellModelFilter, FractionFilter, CellTargetting, classmap_entry="representatives"
):
    """
    Targets all identifiers of certain cell types.
    """

    n: int = config.attr(type=int, default=1)

    @FractionFilter.filter
    def get_targets(self, adapter, simulation, simdata):
        return {
            model: default_rng().choice(len(pop), size=self.n, replace=False)
            for model, pop in super().get_targets(adapter, simulation, simdata)
        }


@config.node
class ByIdTargetting(FractionFilter, CellTargetting, classmap_entry="by_id"):
    """
    Targets all given identifiers.
    """

    ids: dict[str, list[int]] = config.attr(
        type=types.dict(type=types.list(type=int)), required=True
    )

    @FractionFilter.filter
    def get_targets(self, adapter, simulation, simdata):
        by_name = {model.name: model for model in simdata.populations.keys()}
        return {
            model: simdata.populations[model][ids]
            for model_name, ids in self.ids.items()
            if (model := by_name.get(model_name)) is not None
        }


@config.node
class ByLabelTargetting(
    CellModelFilter, FractionFilter, CellTargetting, classmap_entry="by_label"
):
    """
    Targets all given labels.
    """

    labels: list[str] = config.attr(type=types.list(type=str), required=True)

    @FractionFilter.filter
    def get_targets(self, adapter, simulation, simdata):
        return {
            model: simdata.populations[
                simdata.placement[model].get_label_mask(self.labels)
            ]
            for model in super().get_targets(adapter, simulation, simdata).keys()
        }


@config.node
class CylindricalTargetting(
    CellModelFilter, FractionFilter, CellTargetting, classmap_entry="cylinder"
):
    """
    Targets all cells in a cylinder along specified axis.
    """

    origin: np.ndarray[float] = config.attr(type=types.ndarray(shape=(2,), dtype=float))
    """Coordinates of the base of the cylinder for each non main axis"""
    axis: typing.Union[typing.Literal["x"], typing.Literal["y"], typing.Literal["z"]] = (
        config.attr(type=types.in_(["x", "y", "z"]), default="y")
    )
    """Main axis of the cylinder"""
    radius: float = config.attr(type=float, required=True)
    """Radius of the cylinder"""

    @FractionFilter.filter
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
            model: simdata.populations[model][
                np.sum(
                    simdata.placement[model].load_positions()[:, axes] - self.origin**2,
                    axis=1,
                )
                < self.radius**2
            ]
            for model in super().get_targets(adapter, simulation, simdata).keys()
        }


@config.node
class SphericalTargetting(
    CellModelFilter, FractionFilter, CellTargetting, classmap_entry="sphere"
):
    """
    Targets all cells in a sphere.
    """

    origin: list[float] = config.attr(type=types.list(type=float, size=3), required=True)
    radius: float = config.attr(type=float, required=True)

    @FractionFilter.filter
    def get_targets(self, adapter, simulation, simdata):
        """
        Target all or certain cells within a sphere of specified radius.
        """
        return {
            model: simdata.populations[model][
                (
                    np.sum(
                        (simdata.placement[model].load_positions() - self.origin) ** 2,
                        axis=1,
                    )
                    < self.radius**2
                )
            ]
            for model in super().get_targets(adapter, simulation, simdata).keys()
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


@config.node
class LabelTargetting(LocationTargetting, classmap_entry="label"):
    labels = config.list(required=True)

    def get_locations(self, cell):
        locs = [
            loc
            for loc in cell.locations.values()
            if all(l in loc.section.labels for l in self.labels)
        ]
        return locs


@config.node
class BranchLocTargetting(LabelTargetting, classmap_entry="branch"):
    x = config.attr(type=types.fraction(), default=0.5)

    def get_locations(self, cell):
        locations = super().get_locations(cell)
        branches = set()
        selected = []
        for loc in locations:
            if (
                loc._loc[0] not in branches
                and loc.arc(0) <= self.x
                and loc.arc(1) > self.x
            ):
                selected.append(loc)
                branches.add(loc._loc[0])
        return selected


__all__ = [
    "BranchLocTargetting",
    "ByIdTargetting",
    "ByLabelTargetting",
    "CellModelFilter",
    "CellModelTargetting",
    "CellTargetting",
    "ConnectionTargetting",
    "CylindricalTargetting",
    "FractionFilter",
    "LabelTargetting",
    "LocationTargetting",
    "RepresentativesTargetting",
    "SomaTargetting",
    "SphericalTargetting",
    "Targetting",
]
