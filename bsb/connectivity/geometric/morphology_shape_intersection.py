import numpy as np

from ... import config
from ...config import types
from .. import ConnectionStrategy
from .shape_morphology_intersection import _create_geometric_conn_arrays
from .shape_shape_intersection import ShapeHemitype


@config.node
class MorphologyToShapeIntersection(ConnectionStrategy):
    postsynaptic = config.attr(type=ShapeHemitype, required=True)
    affinity = config.attr(type=types.fraction(), required=True, hint=0.1)

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
        post_pos = post_ps.load_positions()[:, [0, 2, 1]]

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

            for post_id, post_coord in enumerate(post_pos):
                post_shapes.translate(post_coord)
                mbb_check = post_shapes.inside_mbox(pre_morpho_coord)
                if np.any(mbb_check):
                    inside_pts = post_shapes.inside_shapes(pre_morpho_coord[mbb_check])
                    if np.any(inside_pts):
                        local_selection = (pre_points_ids[mbb_check])[inside_pts]
                        if self.affinity < 1 and len(local_selection) > 0:
                            nb_sources = np.max(
                                [1, int(np.floor(self.affinity * len(local_selection)))]
                            )
                            chosen_targets = np.random.choice(
                                local_selection.shape[0], nb_sources
                            )
                            local_selection = local_selection[chosen_targets, :]

                        selected_count = len(local_selection)
                        if selected_count > 0:
                            to_connect_pre = np.vstack([to_connect_pre, local_selection])
                            post_tmp = np.full([len(local_selection), 3], -1, dtype=int)
                            post_tmp[:, 0] = post_id
                            to_connect_post = np.vstack([to_connect_post, post_tmp])
                post_shapes.translate(-post_coord)
        # print("Connected", len(pre_pos), "pre cells to", len(post_pos), "post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
