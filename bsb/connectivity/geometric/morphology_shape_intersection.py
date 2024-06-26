import numpy as np

from ... import config
from ...config import types
from .. import ConnectionStrategy
from .shape_morphology_intersection import _create_geometric_conn_arrays
from .shape_shape_intersection import ShapeHemitype


def overlap_boxes(box1_min, box1_max, box2_min, box2_max):
    """
    Check if two minimal bounding box are overlapping.

    :param numpy.ndarray box1_min: 3D point representing the lowest coordinate of the
        minimal bounding box.
    :param numpy.ndarray box1_max: 3D point representing the highest coordinate of the
        minimal bounding box.
    :param numpy.ndarray box2_min: 3D point representing the lowest coordinate of the
        minimal bounding box.
    :param numpy.ndarray box2_max: 3D point representing the highest coordinate of the
        minimal bounding box.
    """
    return np.all((box1_max >= box2_min) & (box2_max >= box1_min))


@config.node
class MorphologyToShapeIntersection(ConnectionStrategy):
    postsynaptic = config.attr(type=ShapeHemitype, required=True)
    affinity = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of apositions to keep over the total number of contact points"""
    pruning_ratio = config.attr(type=types.fraction(), required=True, hint=0.1)
    """Ratio of conections to keep over the total number of apositions"""

    def get_region_of_interest(self, chunk):
        lpre, upre = self.presynaptic._get_rect_ext(tuple(chunk.dimensions))
        post_chunks = self.postsynaptic.get_all_chunks()
        tree = self.postsynaptic._get_shape_boxtree(
            post_chunks,
        )
        pre_mbb = [
            np.concatenate(
                [
                    (lpre + chunk) * chunk.dimensions,
                    np.maximum((upre + chunk), (lpre + chunk) + 1) * chunk.dimensions,
                ]
            )
        ]
        return [post_chunks[i] for i in tree.query(pre_mbb, unique=True)]

    def connect(self, pre, post):
        for pre_ps in pre.placement:
            for post_ps in post.placement:
                self._connect_type(pre_ps, post_ps)

    def _connect_type(self, pre_ps, post_ps):
        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()

        post_shapes = self.postsynaptic.shapes_composition.__copy__()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        morpho_set = pre_ps.load_morphologies()
        pre_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for pre_id, (pre_coord, morpho) in enumerate(zip(pre_pos, pre_morphos)):
            # Get the branches
            branches = morpho.get_branches()

            # Build ids array from the morphology
            pre_points_ids, pre_morpho_coord = _create_geometric_conn_arrays(
                branches, pre_id, pre_coord
            )
            pre_min_mbb, pre_max_mbb = morpho.bounds
            pre_min_mbb += pre_coord
            pre_max_mbb += pre_coord

            tmp_pre_selection = np.full(
                [len(post_pos) * int(len(pre_morpho_coord) * self.affinity), 3],
                -1,
                dtype=int,
            )
            tmp_post_selection = np.full(
                [len(post_pos) * int(len(pre_morpho_coord) * self.affinity), 3],
                -1,
                dtype=int,
            )
            ptr = 0

            for post_id, post_coord in enumerate(post_pos):
                post_shapes.translate(post_coord)
                if overlap_boxes(
                    post_shapes.get_mbb_min(),
                    post_shapes.get_mbb_max(),
                    pre_min_mbb,
                    pre_max_mbb,
                ):
                    mbb_check = post_shapes.inside_mbox(pre_morpho_coord)
                    if np.any(mbb_check):
                        inside_pts = post_shapes.inside_shapes(
                            pre_morpho_coord[mbb_check]
                        )
                        if np.any(inside_pts):
                            local_selection = (pre_points_ids[mbb_check])[inside_pts]
                            if self.affinity < 1 and len(local_selection) > 0:
                                nb_sources = np.max(
                                    [
                                        1,
                                        int(
                                            np.floor(self.affinity * len(local_selection))
                                        ),
                                    ]
                                )
                                chosen_targets = np.random.choice(
                                    local_selection.shape[0], nb_sources
                                )
                                local_selection = local_selection[chosen_targets, :]

                            selected_count = len(local_selection)
                            if selected_count > 0:
                                tmp_pre_selection[ptr : ptr + selected_count, 0] = pre_id
                                tmp_post_selection[ptr : ptr + selected_count, 0] = (
                                    post_id
                                )
                                ptr += selected_count

                post_shapes.translate(-post_coord)

            to_connect_pre = np.vstack([to_connect_pre, tmp_pre_selection[:ptr]])
            to_connect_post = np.vstack([to_connect_post, tmp_post_selection[:ptr]])

        if self.pruning_ratio < 1 and len(to_connect_pre) > 0:
            ids_to_select = np.random.choice(
                len(to_connect_pre),
                int(np.floor(self.pruning_ratio * len(to_connect_pre))),
                replace=False,
            )
            to_connect_pre = to_connect_pre[ids_to_select]
            to_connect_post = to_connect_post[ids_to_select]

        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
