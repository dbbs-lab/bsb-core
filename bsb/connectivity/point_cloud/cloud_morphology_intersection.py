import numpy as np
import itertools
from bsb.connectivity import ConnectionStrategy
from bsb import config
from bsb.connectivity.point_cloud.geometric_shapes import ShapesComposition


@config.node
class CloudToMorphologyIntersection(ConnectionStrategy):
    # Read vars from the configuration file
    affinity = config.attr(type=float, required=True)

    def get_region_of_interest(self, chunk):
        ct = self.postsynaptic.cell_types[0]
        chunks = ct.get_placement_set().get_all_chunks()
        return chunks

    def connect(self, pre, post):
        # pre_type = pre.cell_types[0]
        # post_type = post.cell_types[0]
        for pre_ct, pre_ps in pre.placement.items():
            for post_ct, post_ps in post.placement.items():
                self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()

        cloud_cache = []
        for fn in self.presynaptic.cloud_names:
            cloud = ShapesComposition()
            cloud.load_from_file(fn)
            cloud = cloud.filter_by_labels(self.presynaptic.morphology_labels)
            cloud_cache.append(cloud)

        cloud_choice_id = np.random.randint(
            low=0, high=len(cloud_cache), size=len(pre_pos), dtype=int
        )

        to_connect_pre = np.empty([1, 3], dtype=int)
        to_connect_post = np.empty([1, 3], dtype=int)

        morpho_set = post_ps.load_morphologies()
        post_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for post_id, post_coord, morpho in zip(itertools.count(), post_pos, post_morphos):
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
                pre_cloud = cloud_cache[cloud_choice_id[pre_id]].copy()
                pre_coord[[1, 2]] = pre_coord[[2, 1]]
                pre_cloud.translate(pre_coord)
                local_selection = np.empty([morpho_points, 3])

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

        self.connect_cells(pre_ps, post_ps, to_connect_pre[1:], to_connect_post[1:])
