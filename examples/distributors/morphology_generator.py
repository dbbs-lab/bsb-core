import numpy as np

from bsb import Branch, Morphology, MorphologyGenerator


class TouchTheBottomMorphologies(MorphologyGenerator, classmap_entry="touchdown"):
    def generate(self, positions, morphologies, context):
        return [
            Morphology([Branch([pos, [pos[1], 0, pos[2]]], [1, 1])]) for pos in positions
        ], np.arange(len(positions))
