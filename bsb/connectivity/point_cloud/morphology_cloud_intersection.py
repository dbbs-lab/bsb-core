import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from .cloud_cloud_intersection import CloudHemitype, get_postsyn_chunks


@config.node
class MorphologyToCloudIntersection(ConnectionStrategy):
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
        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()[:, [0, 2, 1]]

        post_cloud = self.postsynaptic.shapes_composition.copy()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        morpho_set = pre_ps.load_morphologies()
        pre_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for pre_id, (pre_coord, morpho) in enumerate(zip(pre_pos, pre_morphos)):
            # Get the branches
            branches = morpho.get_branches()

            # Build ids array from the morphology
            morpho_points = 0
            for b in branches:
                morpho_points += len(b.points)
            pre_points_ids = np.empty([morpho_points, 3], dtype=int)
            pre_morpho_coord = np.empty([morpho_points, 3], dtype=float)
            local_ptr = 0
            for i, b in enumerate(branches):
                pre_points_ids[local_ptr : local_ptr + len(b.points), 0] = pre_id
                pre_points_ids[local_ptr : local_ptr + len(b.points), 1] = i
                pre_points_ids[local_ptr : local_ptr + len(b.points), 2] = np.arange(
                    len(b.points)
                )
                tmp = b.points + pre_coord
                # Swap y and z
                tmp[:, [1, 2]] = tmp[:, [2, 1]]
                pre_morpho_coord[local_ptr : local_ptr + len(b.points)] = tmp
                local_ptr += len(b.points)

            for post_id, post_coord in enumerate(post_pos):
                post_cloud.translate(post_coord)
                mbb_check = post_cloud.inside_mbox(pre_morpho_coord)
                if np.any(mbb_check):
                    inside_pts = post_cloud.inside_shapes(pre_morpho_coord[mbb_check])
                    if np.any(inside_pts):
                        local_selection = (pre_points_ids[mbb_check])[inside_pts]
                        if self.affinity < 1 and len(local_selection) > 0:
                            local_selection = local_selection[
                                np.random.choice(
                                    local_selection.shape[0],
                                    np.max(
                                        [
                                            1,
                                            int(
                                                np.floor(
                                                    self.affinity * len(local_selection)
                                                )
                                            ),
                                        ]
                                    ),
                                ),
                                :,
                            ]

                        selected_count = len(local_selection)
                        if selected_count > 0:
                            to_connect_pre = np.vstack([to_connect_pre, local_selection])
                            post_tmp = np.full([len(local_selection), 3], -1, dtype=int)
                            post_tmp[:, 0] = post_id
                            to_connect_post = np.vstack([to_connect_post, post_tmp])
                post_cloud.translate(-post_coord)
        # print("Connected", len(pre_pos), "pre cells to", len(post_pos), "post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
