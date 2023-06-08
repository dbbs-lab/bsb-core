import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from .cloud_cloud_intersection import CloudHemitype


@config.node
class CloudToMorphologyIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=CloudHemitype)
    affinity = config.attr(type=float, required=True)

    def get_region_of_interest(self, chunk):
        return [
            c
            for ct in self.postsynaptic.cell_types
            for c in ct.get_placement_set().get_all_chunks()
        ]

    def connect(self, pre, post):
        for pre_ct, pre_ps in pre.placement.items():
            for post_ct, post_ps in post.placement.items():
                self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()[:, [0, 2, 1]]
        post_pos = post_ps.load_positions()

        pre_cloud = self.presynaptic.shapes_composition.copy()

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        morpho_set = post_ps.load_morphologies()
        post_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for post_id, (post_coord, morpho) in enumerate(zip(post_pos, post_morphos)):
            # Get the branches
            branches = morpho.get_branches()

            # Build ids array from the morphology
            morpho_points = 0
            for b in branches:
                morpho_points += len(b.points)
            post_points_ids = np.empty([morpho_points, 3], dtype=int)
            post_morpho_coord = np.empty([morpho_points, 3], dtype=float)
            local_ptr = 0
            for i, b in enumerate(branches):
                post_points_ids[local_ptr : local_ptr + len(b.points), 0] = post_id
                post_points_ids[local_ptr : local_ptr + len(b.points), 1] = i
                post_points_ids[local_ptr : local_ptr + len(b.points), 2] = np.arange(
                    len(b.points)
                )
                tmp = b.points + post_coord
                tmp[:, [1, 2]] = tmp[:, [2, 1]]
                post_morpho_coord[local_ptr : local_ptr + len(b.points), :] = tmp
                local_ptr += len(b.points)

            for pre_id, pre_coord in enumerate(pre_pos):
                pre_cloud.translate(pre_coord)
                mbb_check = pre_cloud.inside_mbox(post_morpho_coord)

                if np.any(mbb_check):
                    inside_pts = pre_cloud.inside_shapes(post_morpho_coord[mbb_check])
                    # Find the morpho points inside the cloud
                    if np.any(inside_pts):
                        local_selection = (post_points_ids[mbb_check])[inside_pts]
                        if self.affinity < 1.0 and len(local_selection) > 0:
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
                            to_connect_post = np.vstack(
                                [to_connect_post, local_selection]
                            )
                            pre_tmp = np.full([selected_count, 3], -1, dtype=int)
                            pre_tmp[:, 0] = pre_id
                            to_connect_pre = np.vstack([to_connect_pre, pre_tmp])
                pre_cloud.translate(-pre_coord)
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
