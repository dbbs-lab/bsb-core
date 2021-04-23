import numpy as np, itertools
from .strategy import ConnectionStrategy, TouchingConvergenceDivergence
from ..helpers import DistributionConfiguration
from ..reporting import warn
from ..exceptions import *
from sklearn.neighbors import KDTree


class Convergence(TouchingConvergenceDivergence):
    """
    Implementation of a general convergence connectivity between
    two populations of cells (this does not work with entities)
    """

    def validate(self):
        pass

    def connect(self):
        # Source and target neurons are extracted
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        pre = self.from_cells[from_type.name]
        post = self.to_cells[to_type.name]
        convergence = self.convergence

        pre_post = np.zeros((convergence * len(post), 2))
        for i, neuron in enumerate(post):
            connected_pre = np.random.choice(pre[:, 0], convergence, replace=False)
            range_i = range(i * convergence, (i + 1) * convergence)
            pre_post[range_i, 0] = connected_pre.astype(int)
            pre_post[range_i, 1] = neuron[0]

        self.scaffold.connect_cells(self, pre_post)


class DistanceBased(ConnectionStrategy):
    """
    Create connections using a distance based statistical distribution.
    """

    casts = {
        "distr": DistributionConfiguration.cast,
        "cdf_cutoff": float,
        "pdf_norm": float,
        "max": float,
    }

    defaults = {"cdf_cutoff": 0.95, "max": None, "pdf_norm": 1.0}

    def validate(self):
        print("Distr max?", self.max)
        if self.distr.distribution is None and self.max is None:
            raise ConfigurationError(
                "DistanceBased connectivity requires either a non-uniform"
                + " distribution and a `cdf_cutoff` or a set `max` distance to"
                + " determine the maximum search radius."
            )
        if self.max is None:
            self.max = self.distr.distribution.ppf(self.cdf_cutoff)
            print("Distr max, calculated", self.max)

    def connect(self):
        rng = np.random.default_rng()
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        from_cells = self.from_cells[from_type.name]
        from_ids = from_cells[:, 0]
        to_cells = self.to_cells[to_type.name]
        to_ids = to_cells[:, 0]
        tree = KDTree(to_cells[:, 2:5])
        candidates, distances = tree.query_radius(
            from_cells[:, 2:5], r=self.max, return_distance=True
        )
        n_candidates = sum(map(len, candidates)) - len(candidates)
        rolls = iter(rng.random(n_candidates))
        thresholds = [self.distr.distribution.pdf(d) for d in distances]
        alloc = np.empty((len(from_cells) * len(to_cells), 2))
        ptr = 0
        pdfnorm = self.pdf_norm
        for from_i, (cands, ts) in enumerate(zip(candidates, thresholds)):
            # Candidate iterator, skip first (is cell itself)
            c_iter = iter(cands)
            next(c_iter)
            # Select all candidates who roll under the distance based threshold
            # (probability/threshold >= 1 means all candidates will be selected)
            selected = [
                to_ids[tid] for tid, t in zip(c_iter, ts) if next(rolls) * pdfnorm < t
            ]
            # Create the connections for this from_cell to all selected cells.
            alloc[ptr : ptr + len(selected), 0] = from_ids[from_i]
            alloc[ptr : ptr + len(selected), 1] = selected
            ptr += len(selected)
        self.scaffold.connect_cells(self, alloc[:ptr])


class DegreeAndDistanceBased(ConnectionStrategy):
    """
    Create connections using a distance based statistical distribution.
    """

    casts = {
        "indegree": DistributionConfiguration.cast,
        "outdegree": DistributionConfiguration.cast,
        "distance": DistributionConfiguration.cast,
        "cdf_cutoff": float,
        "pdf_norm": float,
        "max": float,
    }

    defaults = {"cdf_cutoff": 0.95, "max": None, "pdf_norm": 1.0}

    def validate(self):
        print("Distr max?", self.max)
        if self.distr.distribution is None and self.max is None:
            raise ConfigurationError(
                "DegreeAndDistanceBased connectivity requires either a non-uniform"
                + " distribution and a `cdf_cutoff` or a set `max` distance to"
                + " determine the maximum search radius."
            )
        if self.max is None:
            self.max = self.distr.distribution.ppf(self.cdf_cutoff)
            print("Distr max, calculated", self.max)
        indegree = self.indegree.distribution
        max_survivor = indegree.ppf(cdf_cutoff)
        self.survivor_table = np.diff(np.fromiter(map(indegree.cdf, range(max_survivor + 1)))

    def connect(self):
        rng = np.random.default_rng()
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        from_cells = self.from_cells[from_type.name]
        from_ids = from_cells[:, 0]
        to_cells = self.to_cells[to_type.name]
        to_ids = to_cells[:, 0]
        tree = KDTree(to_cells[:, 2:5])
        # Look up all candidate cells within the max search radius, for each cell.
        candidates, distances = tree.query_radius(
            from_cells[:, 2:5], r=self.max, return_distance=True
        )
        # Draw n samples from the outdegree distribution.
        outdegrees = self.outdegree.draw(len(from_cells))
        n_dist_candidates = sum(map(len, candidates)) - len(candidates)
        if outdegrees < n_dist_candidates:
            # Warn in case of funky inputs
            warn("Outdegree exceeds candidates found within search distance.", ConnectivityWarning)
        indegrees = np.zeros(len(to_cells), dtype=int)
        alloc = np.empty((len(from_cells) * len(to_cells), 2))
        ptr = 0
        for i, cand, dist, out in zip(range(len(from_cells)), candidates, distances, outdegrees):
            # Skip self
            cand = cand[1:]
            dist = dist[1:]
            if not len(cand):
                continue
            # Lookup the candidate indegrees
            inds = indegrees[cand]
            # Find the probabilities for transitions of `indegree` to `indegree + 1` and
            # returns 0 for cells already at max indegree; survivors near max indegree
            # should yield close-to-zero probabilities anyway.
            ind_probs = np.where(inds < max_survivor, self.survivor_table, 0)
            # Lookup the distance based probabilities
            dist_probs = self.distance.distribution.pdf(dist)
            probs = ind_probs * dist_probs
            # Choose `outdegree` candidates based on probabilities, unless all candidates
            # are max indegree candidates (all 0 prob), then use equal probabilities.
            targets = rng.choice(cand, size=out, p=probs if sum(probs) else None)
            # Update the target indegrees
            indegrees[targets] += 1
            # Fill in connectivity matrix for this from_cell to its targets.
            alloc[ptr : ptr + len(targets), 0] = from_ids[i]
            alloc[ptr : ptr + len(targets), 1] = to_ids[targets]
            ptr += len(targets)
        self.scaffold.connect_cells(self, alloc[:ptr])


class AllToAll(ConnectionStrategy):
    """
    All to all connectivity between two neural populations
    """

    def validate(self):
        pass

    def connect(self):
        from_type = self.from_cell_types[0]
        to_type = self.to_cell_types[0]
        from_cells = self.from_cells[from_type.name]
        to_cells = self.to_cells[to_type.name]
        l = len(to_cells)
        connections = np.empty([len(from_cells) * l, 2])
        to_cell_ids = to_cells[:, 0]
        for i, from_cell in enumerate(from_cells[:, 0]):
            connections[range(i * l, (i + 1) * l), 0] = from_cell
            connections[range(i * l, (i + 1) * l), 1] = to_cell_ids
        self.scaffold.connect_cells(self, connections)
