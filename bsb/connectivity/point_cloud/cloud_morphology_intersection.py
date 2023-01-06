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
class CloudToMorphologyIntersection(ConnectionStrategy):
    # Read vars from the configuration file
    affinity = config.attr(type=int, required=True)

    def get_region_of_interest(self, chunk):

        ct = self.postsynaptic.cell_types[0]
        chunks = ct.get_placement_set().get_all_chunks()
        

        """cloud = ShapesComposition(voxel_size)
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
        #pre_type = pre.cell_types[0]
        #post_type = post.cell_types[0]
        for pre_ct, pre_ps in pre.placement.items():
            for post_ct, post_ps in post.placement.items():
                self._connect_type(pre_ct, pre_ps, post_ct, post_ps)

    def _connect_type(self, pre_ct, pre_ps, post_ct, post_ps):

        pre_pos = pre_ps.load_positions()
        post_pos = post_ps.load_positions()

        cloud = ShapesComposition()
        cloud.load_from_file(self.presynaptic.cloud_name)
        cloud = cloud.filter_by_labels(self.presynaptic.morphology_labels)

        to_connect_pre = np.empty([1,3],dtype=int)
        to_connect_post = np.empty([1,3],dtype=int)

        morpho_set = post_ps.load_morphologies()
        post_morphos = morpho_set.iter_morphologies(cache=True, hard_cache=True)

        for post_id, post_coord, morpho in zip(itertools.count(), pre_pos, post_morphos):

            #print(post_id, "/", len(post_coord))
            #Get the branches
            branches = morpho.get_branches()
            first_axon_branch_id = branches.index(branches[0])
            
            #Build ids array from the morphology
            morpho_points = 0
            for b in branches:
                morpho_points += len(b.points)
            post_points_ids = np.empty([morpho_points,3],dtype=int)
            post_morpho_coord = np.empty([morpho_points,3],dtype=float)
            local_ptr = 0
            for i,b in enumerate(branches):
                post_points_ids[local_ptr:local_ptr+len(b.points),0] = post_id
                post_points_ids[local_ptr:local_ptr+len(b.points),1] = i
                post_points_ids[local_ptr:local_ptr+len(b.points),2] = np.arange(len(b.points))
                tmp = b.points + post_coord
                tmp[:, [1, 2]] = tmp[:, [2, 1]]
                post_morpho_coord[local_ptr:local_ptr+len(b.points),:]  = tmp
                local_ptr += len(b.points)
            
            """#Find pre minimal bounding box of the morpho
            post_mbb_min = np.min(post_morpho_coord,axis=0)
            post_mbb_max = np.max(post_morpho_coord,axis=0)"""

            for pre_id, pre_coord in enumerate(pre_pos):
                
                pre_cloud = cloud.copy()
                pre_coord[[1, 2]] = pre_coord[[2, 1]]
                pre_cloud.translate(pre_coord)

                local_selection = np.empty([morpho_points,3])

                mbb_check = pre_cloud.inside_mbox(post_morpho_coord)
                #print(mbb_check)
                #print("---------")
                if np.any(mbb_check):
                    print("Found (mbb)")
                    #print(post_coord)
                    #print(post_morpho_coord[mbb_check])
                    inside_pts = pre_cloud.inside_shapes(post_morpho_coord)
                    #Find the morpho points inside the cloud
                    if np.any(inside_pts):
                        print("Found (shapes)")
                        local_selection = post_points_ids[inside_pts]
                        if self.affinity < 1 and len(post_morpho_coord[inside_pts]) > 0:
                            local_selection = local_selection[np.random.choice(local_selection.shape[0], np.max([1, int(np.floor(self.affinity * len(local_selection)))])),:]
                        #print(local_selection)
                        #local_selection = pre_points_ids[inside_pts]
                        
                        selected_count = len(local_selection)
                        if (selected_count > 0):
                            to_connect_post = np.vstack([to_connect_post,local_selection])
                            pre_tmp = np.full([1,3],-1,dtype=int)
                            pre_tmp[:,0] = pre_id
                            to_connect_pre= np.vstack([to_connect_pre,pre_tmp])

        print("Connected", len(pre_pos), "pre cells to",len(post_pos),"post cells.")
        self.connect_cells(pre_ps, post_ps, to_connect_pre[1:], to_connect_post[1:])
