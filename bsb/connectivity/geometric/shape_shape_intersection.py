import numpy as np

from ... import config
from ...config import types
from ...trees import BoxTree
from .. import ConnectionStrategy
from ..strategy import Hemitype
from .geometric_shapes import ShapesComposition


@config.node
class ShapeHemitype(Hemitype):
    """
    Class representing a population of cells to connect with a ConnectionStrategy.
    These cells' morphology is implemented as a ShapesComposition.
    """

    shapes_composition = config.attr(type=ShapesComposition, required=True)
    """
    Composite shape representing the Hemitype.
    """

    def get_mbb(self, chunks, chunk_dimension):
        """
        Get the list of minimal bounding box containing all cells in the `ShapeHemitype`.

        :param chunks: List of chunks containing the cell types
            (see bsb.connectivity.strategy.Hemitype.get_all_chunks)
        :type chunks: List[bsb.storage._chunks.Chunk]
        :param chunk_dimension: Size of a chunk
        :type chunk_dimension: float
        :return: List of bounding boxes in the form [min_x, min_y, min_z, max_x, max_y, max_z]
            for each chunk containing cells.
        :rtype: List[numpy.ndarray[float, float, float, float, float, float]]
        """
        return [
            np.concatenate(
                [
                    self.shapes_composition.get_mbb_min()
                    + np.array(idx_chunk) * chunk_dimension,
                    self.shapes_composition.get_mbb_max()
                    + np.array(idx_chunk) * chunk_dimension,
                ]
            )
            for idx_chunk in chunks
        ]

    def _get_shape_boxtree(self, chunks):
        mbbs = self.get_mbb(chunks, chunks[0].dimensions)
        return BoxTree(mbbs)


@config.node
class ShapeToShapeIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=ShapeHemitype, required=True)
    postsynaptic = config.attr(type=ShapeHemitype, required=True)
    affinity = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of apositions to keep over the total number of contact points"""
    pruning_ratio = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of conections to keep over the total number of apositions"""

    def get_region_of_interest(self, chunk):
        # Filter postsyn chunks that overlap the presyn chunk.
        post_chunks = self.postsynaptic.get_all_chunks()
        tree = self.postsynaptic._get_shape_boxtree(
            post_chunks,
        )
        pre_mbb = self.presynaptic.get_mbb(
            self.presynaptic.get_all_chunks(), chunk.dimensions
        )
        return [post_chunks[i] for i in tree.query(pre_mbb, unique=True)]

    def connect(self, pre, post):
        for pre_ps in pre.placement:
            for post_ps in post.placement:
                self._connect_type(pre_ps.cell_type, pre_ps, post_ps.cell_type, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()

        pre_shapes_cache = self.presynaptic.shapes_composition.__copy__()
        post_shapes_cache = self.postsynaptic.shapes_composition.__copy__()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        for pre_id, pre_coord in enumerate(pre_pos):
            # Generate pre point cloud
            pre_shapes_cache.translate(pre_coord)
            pre_point_cloud = pre_shapes_cache.generate_point_cloud()

            def find_mbb(coords):
                maxima = np.max(coords, axis=0)
                minima = np.min(coords, axis=0)
                return minima, maxima

            def BoxesOverlap(box1min, box1max, box2min, box2max):
                return np.all((box1max >= box2min) & (box2max >= box1min))

            pre_mbb_min = pre_shapes_cache.get_mbb_min()
            pre_mbb_max = pre_shapes_cache.get_mbb_max()

            points_per_cloud = int(len(pre_point_cloud) * self.affinity)
            tmp_pre_selection = np.full(
                [len(post_pos) * int(points_per_cloud), 3], -1, dtype=int
            )
            tmp_post_selection = np.full(
                [len(post_pos) * int(points_per_cloud), 3], -1, dtype=int
            )
            ptr = 0
            for post_id, post_coord in enumerate(post_pos):
                post_shapes_cache.translate(post_coord)
                post_mbb_min = post_shapes_cache.get_mbb_min()
                post_mbb_max = post_shapes_cache.get_mbb_max()
                boxes_overlap = BoxesOverlap(
                    post_mbb_min, post_mbb_max, pre_mbb_min, pre_mbb_max
                )
                if boxes_overlap:
                    # Compare pre and post mbbs
                    inside_mbbox = post_shapes_cache.inside_mbox(pre_point_cloud)
                    if np.any(inside_mbbox):
                        inside_pts = post_shapes_cache.inside_shapes(pre_point_cloud)
                        selected = pre_point_cloud[inside_pts]

                        def sizemod(q, aff):
                            ln = len(q)
                            return int(
                                np.floor(ln * aff) + (np.random.rand() < ((ln * aff) % 1))
                            )

                        selected = selected[
                            np.random.randint(
                                len(selected), size=sizemod(selected, self.affinity)
                            ),
                            :,
                        ]
                        n_synapses = len(selected)
                        if n_synapses > 0:
                            tmp_pre_selection[ptr : ptr + n_synapses, 0] = pre_id
                            tmp_post_selection[ptr : ptr + n_synapses, 0] = post_id
                            ptr += n_synapses
                post_shapes_cache.translate(-post_coord)
            if ptr > 0:
                to_connect_pre = np.vstack([to_connect_pre, tmp_pre_selection[:ptr]])
                to_connect_post = np.vstack([to_connect_post, tmp_post_selection[:ptr]])

            pre_shapes_cache.translate(-pre_coord)

        if self.pruning_ratio < 1 and len(to_connect_pre) > 0:
            ids_to_select = np.random.choice(
                len(to_connect_pre),
                int(np.floor(self.pruning_ratio * len(to_connect_pre))),
                replace=False,
            )
            to_connect_pre = to_connect_pre[ids_to_select]
            to_connect_post = to_connect_post[ids_to_select]

        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
