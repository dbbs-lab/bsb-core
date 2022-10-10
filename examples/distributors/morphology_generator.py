from bsb.placement.distributor import MorphologyGenerator
from bsb.morphologies import Morphology, Branch
import numpy as np


class TouchTheBottomMorphologies(MorphologyGenerator, classmap_entry="touchdown"):
    def generate(self, positions, morphologies, context):
        return [
            Morphology([Branch([pos, [pos[1], 0, pos[2]]], [1, 1])]) for pos in positions
        ], np.arange(len(positions))
