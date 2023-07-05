import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from bsb.connectivity.strategy import Hemitype
from .geometric_shapes import ShapesComposition
from ...config import types
from ...trees import BoxTree


@config.node
class CloudHemitype(Hemitype):
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
        Get the list of minimal bounding box containing all cells in the CloudHemitype.

        :param chunks: List of chunks containing the cell types
            (see bsb.connectivity.strategy.Hemitype.get_all_chunks)
        :type chunks: List[bsb.storage.Chunk]
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

    def _get_cloud_boxtree(self, chunks):
        mbbs = self.get_mbb(chunks, chunks[0].dimensions)
        return BoxTree(mbbs)


@config.node
class CloudToCloudIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=CloudHemitype)
    postsynaptic = config.attr(type=CloudHemitype)
    affinity = config.attr(type=types.fraction(), required=True, hint=0.1)

    def get_region_of_interest(self, chunk):
        # Filter postsyn chunks that overlap the presyn chunk.
        post_chunks = self.postsynaptic.get_all_chunks()
        tree = self.postsynaptic._get_cloud_boxtree(
            post_chunks,
        )
        pre_mbb = self.presynaptic.get_mbb(
            self.presynaptic.get_all_chunks(), chunk.dimensions
        )
        return [post_chunks[i] for i in tree.query(pre_mbb, unique=True)]

    def connect(self, pre, post):
        for pre_ct, pre_ps in pre.placement.items():
            for post_ct, post_ps in post.placement.items():
                self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()[:, [0, 2, 1]]
        post_pos = post_ps.load_positions()[:, [0, 2, 1]]

        pre_cloud_cache = self.presynaptic.shapes_composition.copy()
        post_cloud_cache = self.postsynaptic.shapes_composition.copy()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        for pre_id, pre_coord in enumerate(pre_pos):
            # Generate pre points cloud
            pre_cloud_cache.translate(pre_coord)
            pre_point_cloud = pre_cloud_cache.generate_point_cloud()

            # Find pre minimal bounding box of the morpho
            for post_id, post_coord in enumerate(post_pos):
                post_cloud_cache.translate(post_coord)

                # Compare pre and post mbbs
                inside_mbbox = post_cloud_cache.inside_mbox(pre_point_cloud)
                if np.any(inside_mbbox):
                    inside_pts = post_cloud_cache.inside_shapes(pre_point_cloud)
                    selected = pre_point_cloud[inside_pts]
                    if len(selected) > 0:
                        tmp_pre_selection = np.full([len(selected), 3], -1, dtype=int)
                        tmp_pre_selection[:, 0] = pre_id
                        to_connect_pre = np.vstack([to_connect_pre, tmp_pre_selection])
                        tmp_post_selection = np.full([len(selected), 3], -1, dtype=int)
                        tmp_post_selection[:, 0] = post_id
                        to_connect_post = np.vstack([to_connect_post, tmp_post_selection])
                post_cloud_cache.translate(-post_coord)
            pre_cloud_cache.translate(-pre_coord)
        to_connect_pre = to_connect_pre
        to_connect_post = to_connect_post

        if self.affinity < 1 and len(to_connect_pre) > 0:
            ids_to_select = np.random.choice(
                len(to_connect_pre),
                int(np.floor(self.affinity * len(to_connect_pre))),
                replace=False,
            )
            to_connect_pre = to_connect_pre[ids_to_select]
            to_connect_post = to_connect_post[ids_to_select]

        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
