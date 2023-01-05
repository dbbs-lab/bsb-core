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


@config.node
class CloudToCloudIntersection(ConnectionStrategy):
    # Read vars from the configuration file
    #post_cloud_name = config.attr(type=str, required=True)
    #pre_cloud_name = config.attr(type=str, required=True)

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

        post_cloud = ShapesComposition()
        #post_cloud.load_from_file(self.post_cloud_name)
        post_cloud.load_from_file(self.postsynaptic.cell_types[0].cloud_name)
        
        

        pre_cloud = ShapesComposition()
        #pre_cloud.load_from_file(self.pre_cloud_name)
        pre_cloud.load_from_file(self.presynaptic.cell_types[0].cloud_name)

        to_connect_pre = np.empty([1, 3], dtype=int)
        to_connect_post = np.empty([1, 3], dtype=int)

        for pre_id, pre_coord in enumerate(pre_pos):

            # Generate pre points cloud
            current_pre_cloud = pre_cloud.copy()
            current_pre_cloud.translate(self, pre_coord)
            pre_coord = current_pre_cloud.generate_point_cloud()

            # Find pre minimal bounding box of the morpho
            pre_mbb_min, pre_mbb_max = current_pre_cloud.find_mbb()

            for post_id, post_coord in enumerate(post_pos):
                current_post_cloud = post_cloud.copy()
                current_post_cloud.translate(self, post_coord)

                # Compare pre and post mbbs
                post_mbb_min, post_mbb_max = post_cloud.find_mbb()
                if np.all(pre_mbb_min < post_mbb_min) & np.all(
                    pre_mbb_max < post_mbb_max
                ):
                    # Find the morpho points inside the cloud
                    inside_pts = post_cloud.inside_shapes(pre_coord)
                    selected = pre_coord[inside_pts]
                    if len(selected) > 0:
                        tmp_pre_selection = np.array([pre_id, -1, -1])
                        to_connect_pre = np.vstack([to_connect_pre, tmp_pre_selection])
                        tmp_post_selection = np.array([post_id, -1, -1])
                        to_connect_post = np.vstack([to_connect_pre, tmp_post_selection])

        print("Connected", len(pre_pos), "pre cells to", len(post_pos), "post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre, to_connect_post)
