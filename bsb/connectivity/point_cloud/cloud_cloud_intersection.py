import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from bsb.connectivity.point_cloud.geometric_shapes import ShapesComposition


@config.node
class CloudToCloudIntersection(ConnectionStrategy):
    # Read vars from the configuration file
    # post_cloud_name = config.attr(type=str, required=True)
    # pre_cloud_name = config.attr(type=str, required=True)
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

        pre_cloud_cache = []
        for fn in self.presynaptic.cloud_names:
            cloud = ShapesComposition()
            cloud.load_from_file(fn)
            cloud = cloud.filter_by_labels(self.presynaptic.morphology_labels)
            pre_cloud_cache.append(cloud)

        post_cloud_cache = []
        for fn in self.postsynaptic.cloud_names:
            cloud = ShapesComposition()
            cloud.load_from_file(fn)
            cloud = cloud.filter_by_labels(self.postsynaptic.morphology_labels)
            post_cloud_cache.append(cloud)

        pre_cloud_choice_id = np.random.randint(
            low=0, high=len(pre_cloud_cache), size=len(pre_pos), dtype=int
        )
        post_cloud_choice_id = np.random.randint(
            low=0, high=len(post_cloud_cache), size=len(post_pos), dtype=int
        )

        to_connect_pre = np.empty([1, 3], dtype=int)
        to_connect_post = np.empty([1, 3], dtype=int)

        for pre_id, pre_coord in enumerate(pre_pos):
            # Generate pre points cloud
            current_pre_cloud = pre_cloud_cache[pre_cloud_choice_id[pre_id]].copy()
            tmp_pre_coord = np.copy(pre_coord)
            tmp_pre_coord[[1, 2]] = tmp_pre_coord[[2, 1]]
            current_pre_cloud.translate(tmp_pre_coord)
            pre_point_cloud = current_pre_cloud.generate_point_cloud()

            # Find pre minimal bounding box of the morpho
            pre_mbb_min, pre_mbb_max = current_pre_cloud.find_mbb()

            for post_id, post_coord in enumerate(post_pos):
                current_post_cloud = post_cloud_cache[
                    post_cloud_choice_id[post_id]
                ].copy()
                tmp_post_coord = np.copy(post_coord)
                tmp_post_coord[[1, 2]] = tmp_post_coord[[2, 1]]
                current_post_cloud.translate(tmp_post_coord)

                # Compare pre and post mbbs
                post_mbb_min, post_mbb_max = current_post_cloud.find_mbb()

                inside_mbbox = current_post_cloud.inside_mbox(pre_point_cloud)
                if np.any(inside_mbbox):
                    inside_pts = current_post_cloud.inside_shapes(pre_point_cloud)
                    selected = pre_point_cloud[inside_pts]
                    if len(selected) > 0:
                        tmp_pre_selection = np.full([len(selected), 3], -1, dtype=int)
                        tmp_pre_selection[:, 0] = pre_id
                        to_connect_pre = np.vstack([to_connect_pre, tmp_pre_selection])
                        tmp_post_selection = np.full([len(selected), 3], -1, dtype=int)
                        tmp_post_selection[:, 0] = post_id
                        to_connect_post = np.vstack([to_connect_post, tmp_post_selection])

        to_connect_pre = to_connect_pre[1:]
        to_connect_post = to_connect_post[1:]

        if self.affinity < 1 and len(to_connect_pre) > 0:
            ids_to_select = np.arange(start=0, stop=len(to_connect_pre))
            np.random.shuffle(ids_to_select)
            ids_to_select = ids_to_select[
                0 : np.max(
                    [
                        1,
                        int(np.floor(self.affinity * len(to_connect_pre))),
                    ]
                )
            ]
            to_connect_pre = to_connect_pre[ids_to_select]
            to_connect_post = to_connect_post[ids_to_select]

        # print("Connected", len(pre_pos), "pre cells to", len(post_pos), "post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
