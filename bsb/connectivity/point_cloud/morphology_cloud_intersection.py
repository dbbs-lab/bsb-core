import numpy as np
import itertools
from bsb.connectivity import ConnectionStrategy
from bsb.storage import Chunk
from bsb import config
from scipy.stats.distributions import truncexpon
from bsb.morphologies import Morphology
from bsb.storage.interfaces import ConnectivitySet as IConnectivitySet
from bsb.connectivity.strategy import Hemitype
from bsb.connectivity.strategy import HemitypeCollection
from bsb.connectivity.point_cloud.geometric_shapes import ShapesComposition


@config.node
class MorphologyToCloudIntersection(ConnectionStrategy):
    # Read vars from the configuration file
    affinity = config.attr(type=float, required=True)

    def get_region_of_interest(self, chunk):

        ct = self.postsynaptic.cell_types[0]
        chunks = ct.get_placement_set().get_all_chunks()

        """
        cloud = ShapesComposition(voxel_size)
        cloud.load_from_file(cloud_name)  
        mbb_min, mbb_max = cloud.find_mbb()
        
        selected_chunks = []

        # Look for chunks inside the mbb
        #inside = (c[:,0]>mbb_min[0]) & (c[:,1]>mbb_min[1]) & (c[:,2]>mbb_min[2]) & (c[:,0]<mbb_max[0]) & (c[:,1]<mbb_max[1]) & (c[:,2]<mbb_max[2])
        for c in chunks:    
            #inside = (c[0]>mbb_min[0]) & (c[1]>mbb_min[1]) & (c[2]>mbb_min[2]) & (c[0]<mbb_max[0]) & (c[1]<mbb_max[1]) & (c[2]<mbb_max[2])
            inside = (c[0]>mbb_min[0]-c.dimensions[0]) & (c[1]>mbb_min[1]-c.dimensions[1]) & (c[2]>mbb_min[2]-c.dimensions[2]) & (c[0]<mbb_max[0]+c.dimensions[0]) & (c[1]<mbb_max[1]+c.dimensions[1]) & (c[2]<mbb_max[2]+c.dimensions[2])
            if (inside == True):
                selected_chunks.append(Chunk([c[0], c[1], c[2]], chunk.dimensions))
        """

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
        for fn in self.postsynaptic.cloud_names:
            print(fn)
            cloud = ShapesComposition()
            cloud.load_from_file(fn)
            cloud = cloud.filter_by_labels(self.postsynaptic.morphology_labels)
            cloud_cache.append(cloud)

        cloud_choice_id = np.random.randint(
            low=0, high=len(cloud_cache), size=len(post_pos), dtype=int
        )

        to_connect_pre = np.empty([1, 3], dtype=int)
        to_connect_post = np.empty([1, 3], dtype=int)

        morpho_set = pre_ps.load_morphologies()
        pre_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for pre_id, pre_coord, morpho in zip(itertools.count(), pre_pos, pre_morphos):

            # Get the branches
            branches = morpho.get_branches()
            first_axon_branch_id = branches.index(branches[0])

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

                post_cloud = cloud_cache[cloud_choice_id[post_id]].copy()
                # Swap y and z
                post_coord[[1, 2]] = post_coord[[2, 1]]
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

        # print("Connected", len(pre_pos), "pre cells to", len(post_pos), "post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre[1:], to_connect_post[1:])
