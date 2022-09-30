from bsb.placement.distributor import MorphologyDistributor
import numpy as np
from scipy.stats.distributions import norm


class SmallerTopMorphologies(MorphologyDistributor, classmap_entry="small_top"):
    def distribute(self, positions, morphologies, context):
        # Get the maximum Y coordinate of all the partitions boundaries
        top_of_layers = np.maximum([p.data.mdc[1] for p in context.partitions])
        depths = top_of_layers - positions[:, 1]
        # Get all the heights of the morphologies, by peeking into the morphology metadata
        msizes = [
            loader.get_meta()["mdc"][1] - loader.get_meta()["ldc"][1]
            for loader in morphologies
        ]
        # Pick deeper positions for bigger morphologies.
        weights = np.column_stack(
            [norm(loc=size, scale=20).pdf(depths) for size in msizes]
        )
        # The columns are the morphology ids, so make an arr from 0 to n morphologies.
        picker = np.arange(weights.shape[1])
        # An array to store the picked weights
        picked = np.empty(weights.shape[0], dtype=int)
        rng = np.default_rng()
        for i, p in enumerate(weights):
            # Pick a value from 0 to n, based on the weights.
            picked[i] = rng.choice(picker, p=p)
        # Return the picked morphologies for each position.
        return picked
