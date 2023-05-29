from functools import cached_property

import numpy as np
from bsb.connectivity import ConnectionStrategy
from bsb import config
from bsb.connectivity.strategy import Hemitype
from .geometric_shapes import ShapeCompositionDependencyNode


@config.node
class CloudHemitype(Hemitype):
    _shape_compositions = config.list(type=ShapeCompositionDependencyNode)

    @cached_property
    def shape_compositions(self):
        result = []
        for sc in self._shape_compositions:
            result.append(sc.load_object())
            result[-1].filter_by_labels(self.presynaptic.morphology_labels)
        return result


@config.node
class CloudToCloudIntersection(ConnectionStrategy):
    presynaptic = config.attr(type=CloudHemitype)
    postsynaptic = config.attr(type=CloudHemitype)
    affinity = config.attr(type=float, required=True)

    def get_region_of_interest(self, chunk):
        ct = self.postsynaptic.cell_types[0]
        chunks = ct.get_placement_set().get_all_chunks()
        return chunks

    def connect(self, pre, post):
        for pre_ct, pre_ps in pre.placement.items():
            for post_ct, post_ps in post.placement.items():
                self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):
        pre_pos = pre_ps.load_positions()[:, [0, 2, 1]]
        post_pos = post_ps.load_positions()[:, [0, 2, 1]]

        pre_cloud_cache = self.presynaptic.shape_compositions
        pre_cloud_cache = np.array(pre_cloud_cache)[
            np.random.randint(
                low=0, high=len(pre_cloud_cache), size=len(pre_pos), dtype=int
            )
        ]
        post_cloud_cache = self.postsynaptic.shape_compositions
        post_cloud_cache = np.array([post_cl.copy() for post_cl in post_cloud_cache])[
            np.random.randint(
                low=0, high=len(post_cloud_cache), size=len(post_pos), dtype=int
            )
        ]

        to_connect_pre = np.empty([0, 3], dtype=int)
        to_connect_post = np.empty([0, 3], dtype=int)

        for pre_id, pre_coord in enumerate(pre_pos):
            # Generate pre points cloud
            current_pre_cloud = pre_cloud_cache[pre_id]
            current_pre_cloud.translate(pre_coord)
            pre_point_cloud = current_pre_cloud.generate_point_cloud()

            # Find pre minimal bounding box of the morpho
            for post_id, post_coord in enumerate(post_pos):
                current_post_cloud = post_cloud_cache[post_id]
                current_post_cloud.translate(post_coord)

                # Compare pre and post mbbs
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
                current_post_cloud.translate(-post_coord)
            current_pre_cloud.translate(-pre_coord)
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
