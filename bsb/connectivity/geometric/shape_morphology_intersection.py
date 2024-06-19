import numpy as np

from ... import config
from ...config import types
from .. import ConnectionStrategy
from .shape_shape_intersection import ShapeHemitype


def _create_geometric_conn_arrays(branches, ids, coord):
    morpho_points = 0
    for b in branches:
        morpho_points += len(b.points)
    points_ids = np.empty([morpho_points, 3], dtype=int)
    morpho_coord = np.empty([morpho_points, 3], dtype=float)
    local_ptr = 0
    for i, b in enumerate(branches):
        points_ids[local_ptr : local_ptr + len(b.points), 0] = ids
        points_ids[local_ptr : local_ptr + len(b.points), 1] = i
        points_ids[local_ptr : local_ptr + len(b.points), 2] = np.arange(len(b.points))
        tmp = b.points + coord
        morpho_coord[local_ptr : local_ptr + len(b.points), :] = tmp
        local_ptr += len(b.points)
    return points_ids, morpho_coord


@config.node
class ShapeToMorphologyIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=ShapeHemitype, required=True)
    affinity = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of apositions to keep over the total number of contact points"""
    pruning_ratio = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of conections to keep over the total number of apositions"""

    def get_region_of_interest(self, chunk):
        lpost, upost = self.postsynaptic._get_rect_ext(tuple(chunk.dimensions))
        pre_chunks = self.presynaptic.get_all_chunks()
        tree = self.presynaptic._get_shape_boxtree(
            pre_chunks,
        )
        post_mbb = [
            np.concatenate(
                [
                    (lpost + chunk) * chunk.dimensions,
                    np.maximum((upost + chunk), (lpost + chunk) + 1) * chunk.dimensions,
                ]
            )
        ]

        return [pre_chunks[i] for i in tree.query(post_mbb, unique=True)]

    def connect(self, pre, post):
        for pre_ps in pre.placement:
            for post_ps in post.placement:
                self._connect_type(pre_ps.cell_type, pre_ps, post_ps.cell_type, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()

        pre_shapes = self.presynaptic.shapes_composition.__copy__()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        morpho_set = post_ps.load_morphologies()
        post_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for post_id, (post_coord, morpho) in enumerate(zip(post_pos, post_morphos)):
            # Get the branches
            branches = morpho.get_branches()

            # Build ids array from the morphology
            post_points_ids, post_morpho_coord = _create_geometric_conn_arrays(
                branches, post_id, post_coord
            )

            for pre_id, pre_coord in enumerate(pre_pos):
                pre_shapes.translate(pre_coord)
                mbb_check = pre_shapes.inside_mbox(post_morpho_coord)

                if np.any(mbb_check):
                    inside_pts = pre_shapes.inside_shapes(post_morpho_coord[mbb_check])
                    # Find the morpho points inside the postsyn geometric shapes
                    if np.any(inside_pts):
                        local_selection = (post_points_ids[mbb_check])[inside_pts]
                        if self.affinity < 1.0 and len(local_selection) > 0:
                            nb_targets = np.max(
                                [1, int(np.floor(self.affinity * len(local_selection)))]
                            )
                            chosen_targets = np.random.choice(
                                local_selection.shape[0], nb_targets
                            )
                            local_selection = local_selection[chosen_targets, :]
                        selected_count = len(local_selection)
                        if selected_count > 0:
                            to_connect_post = np.vstack(
                                [to_connect_post, local_selection]
                            )
                            pre_tmp = np.full([selected_count, 3], -1, dtype=int)
                            pre_tmp[:, 0] = pre_id
                            to_connect_pre = np.vstack([to_connect_pre, pre_tmp])
                pre_shapes.translate(-pre_coord)
        if self.pruning_ratio < 1 and len(to_connect_pre) > 0:
            ids_to_select = np.random.choice(
                len(to_connect_pre),
                int(np.floor(self.pruning_ratio * len(to_connect_pre))),
                replace=False,
            )
            to_connect_pre = to_connect_pre[ids_to_select]
            to_connect_post = to_connect_post[ids_to_select]
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
