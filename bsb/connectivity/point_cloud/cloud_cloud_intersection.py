import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from bsb.connectivity.strategy import Hemitype
from .geometric_shapes import ShapesComposition
from ...trees import BoxTree


@config.node
class CloudHemitype(Hemitype):
    shapes_composition = config.attr(type=ShapesComposition, required=True)
    """
    Composite shape representing the Hemitype.
    """


def get_postsyn_chunks(presyn_chunk, post_cell_types, post_shapes_composition):
    """
    Returns the list of chunks overlapping with the postsynaptic point clouds, based on a
    chunk containing the presynaptic neurons .

    :param presyn_chunk: Presynaptic chunk
    :type presyn_chunk: bsb.storage.Chunk
    :param post_cell_types: Postsynaptic cell types
    :type post_cell_types: List[bsb.cell_types.CellType]
    :param post_shapes_composition: Composite shape representing the postsynaptic neuron.
    :type post_shapes_composition: ShapesComposition
    :returns: List of postsynaptic chunks
    :rtype: List[bsb.storage.Chunk]
    """
    chunks = [
        c for ct in post_cell_types for c in ct.get_placement_set().get_all_chunks()
    ]
    tree = BoxTree(
        [
            np.concatenate(
                [
                    post_shapes_composition.get_mbb_min() + np.array(pre_coord),
                    post_shapes_composition.get_mbb_max() + np.array(pre_coord),
                ]
            )
            for pre_coord in chunks
        ]
    )
    return [
        chunks[j]
        for i in tree.query(
            [
                np.concatenate(
                    [
                        np.array(presyn_chunk),
                        np.array(presyn_chunk) + presyn_chunk.dimensions,
                    ]
                )
            ]
        )
        for j in i
    ]


@config.node
class CloudToCloudIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=CloudHemitype)
    postsynaptic = config.attr(type=CloudHemitype)
    affinity = config.attr(type=float, required=True)

    def get_region_of_interest(self, chunk):
        return get_postsyn_chunks(
            chunk, self.postsynaptic.cell_types, self.postsynaptic.shapes_composition
        )

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
