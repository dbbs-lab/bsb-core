import abc
import itertools

from bsb import config
from bsb.config import types
from bsb.simulation.cell import CellModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...morphologies import MorphologySet


@config.dynamic(
    attr_name="model_strategy", required=False, default="arborize", auto_classmap=True
)
class NeuronCell(CellModel):
    def create_instances(self, count, pos, morpho: "MorphologySet", rot, additional):
        def dictzip():
            yield from (
                dict(zip(additional.keys(), values[:-1]))
                for values in itertools.zip_longest(
                    *additional.values(), itertools.repeat(count)
                )
            )

        pos, morpho, rot = (
            iter(pos),
            iter(morpho),
            iter(rot),
        )
        additer = dictzip()
        return [
            self.create(i, next(pos), next(morpho), next(rot), next(additer))
            for i in range(count)
        ]

    def create(self, id, pos, morpho, rot, additional):
        raise NotImplementedError("Cell models should implement the `create` method.")


@config.node
class ArborizedModel(NeuronCell, classmap_entry="arborize"):
    model = config.attr(
        type=types.class_(), required=lambda s: not ("relay" in s and s["relay"])
    )
    _schematics = {}

    def create(self, id, pos, morpho, rot, additional):
        from arborize import bsb_schematic, neuron_build

        self.model.use_defaults = True
        schematic = bsb_schematic(morpho, self.model)
        return neuron_build(schematic)
