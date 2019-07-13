import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import distance
from scaffold_params import *


def adapt_thick_coords(sublayers_roof, cell_radius):
	''' Given y-lims of each sublayer, return range of possible positions.
	subalayers_roof: y-lims of sublayers
	cell_radius: radius of the cell'''
	roof_padded = np.roll(sublayers_roof, -1)
	return np.column_stack([sublayers_roof,roof_padded])[:-1]+np.array([cell_radius, -cell_radius])

def sublayer_partitioning(layer, cell_type, volume_base_size, *args):
	'''	layer: string; which layer (granular, purkinje, molecular),
		cell_type: string; which cell
		volume_base_size; 1x2 array'''
	layer_thick = layers_thick[layer]
	if layer == 'molecular':
		layer_thick /= 2.0
	full_volume = layer_thick*volume_base_size[0]*volume_base_size[1]
	for v in args:
		full_volume -= v
	cell_in_volume = full_volume*cells_density[cell_type]
	# cell_in_volume = int((layer_thick*
						# volume_base_size[0]*
						# volume_base_size[1]-v_occupied)*
						# cells_density[cell_type])
	vol_for_cell = 1/cells_density[cell_type]
	vol_for_cell_radius = (3.0/4/np.pi*vol_for_cell)**(1/3.0)
	cell_eps = (vol_for_cell_radius - cells_radius[cell_type])
	cell_eps = np.array([cell_eps, cell_eps])
	cell_sublayers = np.round(layer_thick/(1.5*vol_for_cell_radius))
	height_cell_sublayer = layer_thick/cell_sublayers
	cell_sublayers_roof = np.linspace(height_cell_sublayer, layer_thick, cell_sublayers)
	cell_sublayers_roof = np.insert(cell_sublayers_roof, 0, 0)
	cell_height_placement = adapt_thick_coords(cell_sublayers_roof, cells_radius[cell_type])
	ncell_per_sublayer = np.round(cell_in_volume/cell_sublayers)
	cell_bounds = np.column_stack([np.zeros(2)+cells_radius[cell_type], volume_base_size-cells_radius[cell_type]])

	return cell_in_volume, cell_eps, cell_sublayers, cell_height_placement, ncell_per_sublayer, cell_bounds

def volume_occupied(cell_in_volume, radius):
	return cell_in_volume*4.0/3*np.pi*(radius)**3.0

def compute_circle(center, radius, n_samples=50):
	''' Create circle n_samples starting from given
	center and radius.'''
	nodes = np.linspace(0,2*np.pi,n_samples, endpoint=False)
	x, y = np.sin(nodes)*radius+center[0], np.cos(nodes)*radius+center[1]
	return np.column_stack([x,y])

def linear_project(center, cell, eps):
	''' Linear projections of points on a circle;
	center: center of circle
	cell: radius
	eps: random positive number '''
	return (cell-(center-cell)) + (np.sign(cell-center)*eps)

def rec_intersection(*args):
	''' Intersection of 2 or more arrays (using recursion)'''
	if len(args) == 2:
		return np.intersect1d(args[0], args[1])
	return rec_intersection(np.intersect1d(args[0], args[1]), args[2::])

def purkinje_placement(pc_extension_dend_tree, pc_in_volume):
	'''Special case for placement; kept separated from 'cells_placement' function
		because requires less/different arguments and exploits different algorithm'''
	d = 2 * cells_radius['purkinje']
	eps = (1/2.0)*(-(d+pc_extension_dend_tree)+((d+pc_extension_dend_tree)**(2)-4*((d*pc_extension_dend_tree)-(volume_base_size[0]*volume_base_size[1]/pc_in_volume)))**(1.0/2))
	subl_z = np.linspace(cells_radius['purkinje'],volume_base_size[1]-cells_radius['purkinje']-(eps/2), volume_base_size[1]/(d+eps))

	offset = 0
	delta = subl_z.shape[0]/((pc_extension_dend_tree/2)-1)
	npc = 0

	for i in np.arange(subl_z.shape[0]):

		x = np.arange((pc_extension_dend_tree/2.)+offset+cells_radius['purkinje'],volume_base_size[0]-cells_radius['purkinje'],pc_extension_dend_tree-1)

		z = np.zeros((x.shape[0]))
		for cont in np.arange(x.shape[0]):
			y = np.random.uniform(cells_radius['purkinje'], layers_thick['purkinje']-cells_radius['purkinje'], x.shape[0])
			z[cont] = subl_z[i] + (eps/2)*np.random.rand()
		pos = np.column_stack([x, y, z])
		final_cell_positions['purkinje'].append(pos)

		offset += delta*5.*np.random.rand()
		npc += x.shape[0]

	matrix_reframe = np.empty((1,3))
	placement_stats['purkinje']['total_n_pc'] = npc

	for subl in final_cell_positions['purkinje']:
		matrix_reframe = np.concatenate((matrix_reframe, subl), axis=0)
	final_cell_positions['purkinje'] = matrix_reframe[1::]
	placement_stats['purkinje']['number_of_cells'].append(final_cell_positions['purkinje'].shape[0])


def define_bounds(possible_points, cell_bounds):
	x_mask = (possible_points[:,0].__ge__(cell_bounds[0,0])) & (possible_points[:,0].__le__(cell_bounds[0,1]))
	y_mask = (possible_points[:,1].__ge__(cell_bounds[1,0])) & (possible_points[:,1].__le__(cell_bounds[1,1]))
	return x_mask, y_mask


def cells_placement(cell_sublayers, volume_base_size, cell_type, eps, cell_height_placement, ncell_per_sublayer, cell_bounds):
	if cell_type == 'granule':
		min_mult, max_mult = 2., 2.
	else:
		min_mult, max_mult = 3., 5.

	for subl in np.arange(cell_sublayers):
		subl = int(subl)
		cell_positions = np.array((np.random.uniform(0, volume_base_size[0]),
									np.random.uniform(cell_height_placement[subl,0],cell_height_placement[subl,1]),
									np.random.uniform(0, volume_base_size[1])))

		positions_list = np.array([cell_positions])
		## Place the first cell and generate the second one; NB: should add a check to
		## verify that the randomly selected position is not occupied by a different cell type
		# For Soma and possible points calcs, we take into account only planar coordinates
		center = positions_list[0][[0,2]] # First and Third columns
		# Compute cell soma limits
		cell_soma = compute_circle(center, cells_radius[cell_type])
		# Define random (bounded) epsilon value for minimal distance between two cells
		rnd_eps = np.tile(np.random.uniform(eps[0]*(min_mult/4), eps[0]*(max_mult/4)),2)
		# Create new possible centers for next cell
		possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
		# Constrain their possible positions considering volume (plane) boundaries
		x_mask, y_mask = define_bounds(possible_points, cell_bounds)
		possible_points = possible_points[x_mask & y_mask]
		# If there are no possible points, force the cell position to be in the middle of surface
		if possible_points.shape[0] == 0:
			cell_positions = np.array([volume_base_size[0] / 2., np.random.uniform(cell_height_placement[subl,0],cell_height_placement[subl,1]), volume_base_size[0] / 2.])
			cell_soma = compute_circle(center, cells_radius[cell_type])
			possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
			x_mask, y_mask = define_bounds(possible_points, cell_bounds)
			possible_points = possible_points[x_mask & y_mask]
		# Add third coordinate to all possible points
		possible_points = np.insert(possible_points, 1, np.random.uniform(cell_height_placement[subl, 0], cell_height_placement[subl,1], possible_points.shape[0]), axis=1)
		# Randomly select one of possible points
		new_point = possible_points[np.random.randint(possible_points.shape[0])]
		# Add the new point to list of cells positions
		positions_list= np.vstack([positions_list, new_point])
		# History of good possible points still available
		good_points_store = [possible_points]
		# History of 'dead-ends' points
		bad_points = []

		# Place the rest of the cells for the selected sublayer
		for i in np.arange(1, ncell_per_sublayer):
			i = int(i)
			# Create soma as a circle:
			# start from the center of previously fixed cell
			center = positions_list[-1][[0,2]]
			# Sample n_samples points along the circle surrounding cell center
			cell_soma = compute_circle(center, cells_radius[cell_type])
			# Recalc random epsilon and use it to define min distance from current cell
			# for possible new cells
			rnd_eps = np.tile(np.random.uniform(eps[0]*(min_mult/4), eps[0]*(max_mult/4)),2)
			inter_cell_soma_dist = cells_radius[cell_type]*2+rnd_eps[0]
			possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
			x_mask, y_mask = define_bounds(possible_points, cell_bounds)
			possible_points = possible_points[x_mask & y_mask]
			if possible_points.shape[0] == 0:
				print ("Can't place cells because of volume boundaries")
				break
			# For each candidate, calculate distance from the centers of all the (already placed) cells
			# This comparison is performed ONLY along planar coordinates
			distance_from_centers = distance.cdist(possible_points, positions_list[:,[0,2]])
			# Associate a third dimension
			full_coords = np.insert(possible_points, 1, np.random.uniform(cell_height_placement[subl, 0], cell_height_placement[subl,1], possible_points.shape[0]), axis=1)
			# Check if any of candidate points is placed at acceptable distance from all of the other cells.
			good_idx = list(np.where(np.sum(distance_from_centers.__ge__(inter_cell_soma_dist), axis=1)==distance_from_centers.shape[1])[0])
			if cell_type == 'glomerulus':
				# If the cell type is Glomerulus, we should take into account GoC positions
				inter_glomgoc_dist = cells_radius[cell_type]+cells_radius['golgi']
				distance_from_golgi = distance.cdist(full_coords, final_cell_positions['golgi'])
				good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
				good_idx = rec_intersection(good_idx, good_from_goc)
			if cell_type == 'granule':
				# If the cell type is Granule, we should take into account GoC and Gloms positions
				inter_grcgoc_dist = cells_radius[cell_type]+cells_radius['golgi']
				inter_grcglom_dist = cells_radius[cell_type]+cells_radius['glomerulus']
				distance_from_golgi = distance.cdist(full_coords, final_cell_positions['golgi'])
				distance_from_gloms = distance.cdist(full_coords, final_cell_positions['glomerulus'])
				good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
				good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
				good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
			if len(good_idx) == 0:
				# If we don't find any possible candidate, let's start from the first cell and see
				# if we can find new candidates from previous cells options
				for j in range(len(good_points_store)):
					possible_points = good_points_store[j][:,[0,2]]
					cand_dist = distance.cdist(possible_points, positions_list[:,[0,2]])
					full_coords = good_points_store[j]
					rnd_eps = np.tile(np.random.uniform(eps[0]*(min_mult/4), eps[0]*(max_mult/4)),2)
					inter_cell_soma_dist = cells_radius[cell_type]*2+rnd_eps[0]
					good_idx = list(np.where(np.sum(cand_dist.__ge__(inter_cell_soma_dist), axis=1)==cand_dist.shape[1])[0])
					if cell_type == 'glomerulus':
						distance_from_golgi = distance.cdist(full_coords, final_cell_positions['golgi'])
						good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
						good_idx = rec_intersection(good_idx, good_from_goc)
					if cell_type == 'granule':
						distance_from_golgi = distance.cdist(full_coords, final_cell_positions['golgi'])
						distance_from_gloms = distance.cdist(full_coords, final_cell_positions['glomerulus'])
						good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
						good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
						good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
					if len(good_idx) > 0:
						## If there is at least one good candidate, select one randomly
						new_point_idx = random.sample(list(good_idx), 1)[0]
						center = good_points_store[j][new_point_idx]
						cell_soma = compute_circle(center[[0,2]], cells_radius[cell_type])
						positions_list= np.vstack([positions_list, center])
						#print( "Go back to main loop")
						break
					else:
						bad_points.append(j)
				# If we can't find a good point, the loop must be stopped
				if len(good_idx) == 0:
					print( "Finished after: ", i, " iters")
					break
			else:
				# If there is at least one good candidate, select one randomly
				new_point_idx = random.sample(list(good_idx), 1)[0]
				positions_list= np.vstack([positions_list, full_coords[new_point_idx]])

				# Keep track of good candidates for each cell
				good_points_store = [good_points_store[i] for i in range(len(good_points_store)) if i not in bad_points]
				good_points_store.append(full_coords[good_idx])
				bad_points = []

		final_cell_positions[cell_type].append(positions_list)
		placement_stats[cell_type]['number_of_cells'].append(positions_list.shape[0])
		print( "{} sublayer number {} out of {} filled".format(cell_type, subl+1, cell_sublayers))

	matrix_reframe = np.empty((1,3))
	for subl in final_cell_positions[cell_type]:
		matrix_reframe = np.concatenate((matrix_reframe, subl), axis=0)
	final_cell_positions[cell_type] = matrix_reframe[1::]

def adapt_positions():
	if len(final_cell_positions['purkinje']) != 0:
		final_cell_positions['purkinje'][:,1] += layers_thick['granular']
	if len(final_cell_positions['basket']) != 0:
		final_cell_positions['basket'][:,1] += layers_thick['granular']+layers_thick['purkinje']
	if len(final_cell_positions['stellate']) != 0:
		final_cell_positions['stellate'][:,1] += layers_thick['granular']+layers_thick['purkinje'] + layers_thick['molecular']/2.
	if len(final_cell_positions['dcn']) != 0:
		final_cell_positions['dcn'][:,1] -= layers_thick['dcn']
		final_cell_positions['dcn'][:,[0,2]] = final_cell_positions['dcn'][:,[0,2]] + dcn_volume[0]/2.

	dcn_thick = layers_thick['dcn']
	for key, val in final_cell_positions.items():
		val[:,1] += dcn_thick











# CONNECTIVITY FUNCTIONS

# GRANULAR LAYER

# Connectivity between glomeruli and GrC dendrites
def connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom, glom_grc):

	for i in granules:	# for all granules: calculate which glomeruli can be connected, then choose 4 of them

		# find all glomeruli at a maximum distance of 40micron
		volume_matrix = (((glomeruli[:,2]-i[2])**2)+((glomeruli[:,3]-i[3])**2)+((glomeruli[:,4]-i[4])**2)-(dend_len**2)).__le__(0)
		good_gloms = np.where(volume_matrix==True)[0]	# indexes of glomeruli that can potentially be connected

		if (len(good_gloms))>n_conn_glom:
			gloms_distance = np.sqrt((glomeruli[good_gloms,2]-i[2])**2+(glomeruli[good_gloms,3]-i[3])**2+(glomeruli[good_gloms,4]-i[4])**2)
			dist_matrix = np.zeros((len(good_gloms), 2))
			dist_matrix[:,0] = good_gloms + first_glomerulus
			dist_matrix[:,1] = gloms_distance
			sc_dist = dist_matrix[dist_matrix[:,1].argsort()]	# sorting of the resulting vector on the distances
			connected_f = sc_dist[0:n_conn_glom,0]
			connected_dist = sc_dist[0:n_conn_glom,1]
			connected_provv = connected_f.astype(int)
			connected_gloms = connected_provv

			# construction of the output matrix: the first column has the  glomerulus index, while the second column has the connected granule index
			matrix = np.zeros((n_conn_glom, 3))
			matrix[:,1] = i[0]
			matrix[:,0] = connected_gloms
			matrix[:,2] = gloms_distance[0:4]
			glom_grc = np.vstack((glom_grc, matrix))

		else:
			gloms_distance = np.sqrt((glomeruli[good_gloms,2]-i[2])**2+(glomeruli[good_gloms,3]-i[3])**2+(glomeruli[good_gloms,4]-i[4])**2)
			connected_gloms = good_gloms + first_glomerulus

			matrix = np.zeros((len(connected_gloms), 3))
			matrix[:,1] = i[0]
			matrix[:,0] = connected_gloms
			matrix[:,2] = gloms_distance
			glom_grc = np.vstack((glom_grc, matrix))

	glom_grc = glom_grc[1:-1,:]

	return glom_grc



# Connectivity between glomeruli and basolateral dendrites of GoCs
def connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol, glom_bd):

	# for all Golgi cells: calculate which glomeruli fall into the volume of GoC basolateral dendrites, then choose 40 of them for the connection and delete them from successive computations, since 1 axon is connected to 1 GoC
	for i in golgicells:

		volume_matrix = (((glomeruli[:,2]-i[2])**2)+((glomeruli[:,3]-i[3])**2)+((glomeruli[:,4]-i[4])**2)-(r_goc_vol**2)).__le__(0) & (glomeruli[:,3]).__le__(i[3])
		good_gloms = np.where(volume_matrix==True)[0]	# finds indexes of granules that can potentially be connected
		connected_gloms = good_gloms + first_glomerulus

		# construction of the output matrix: the first column has the index of the connected glomerulus, while the second column has the Golgi cell index
		matrix = np.zeros((len(good_gloms), 3))
		matrix[:,1] = i[0]
		matrix[:,0] = connected_gloms
		matrix[:,2] = np.sqrt((glomeruli[good_gloms,2]-i[2])**2 + (glomeruli[good_gloms,3]-i[3])**2 + (glomeruli[good_gloms,4]-i[4])**2)
		glom_bd = np.vstack((glom_bd, matrix))

	glom_bd = glom_bd[1:-1,:]

	return glom_bd


# Connectivity between GoCs and glomeruli
def connectome_goc_glom(first_glomerulus, glomeruli, golgicells, GoCaxon_x, GoCaxon_y, GoCaxon_z, r_glom, n_conn_goc, OoB_value, axonGOC_glom):

	new_glomeruli = np.copy(glomeruli)
	new_golgicells = np.random.permutation(golgicells)

	# for all Golgi cells: calculate which glomeruli fall into the area of GoC axon, then choose 40 of them for the connection and delete them from successive computations, since 1 glomerulus must be connected to only 1 GoC
	for i in new_golgicells:

		idx = 1

		bool_matrix = np.zeros((new_glomeruli.shape[0], 3))		# matrix initialization
		# glomerulus falls into the z range of values?
		bool_matrix[:,0] = (((new_glomeruli[:,2]+r_glom).__ge__(i[2]-GoCaxon_x/2.)) & ((new_glomeruli[:,2]-r_glom).__le__(i[2]+GoCaxon_x/2.)))
		# glomerulus falls into the y range of values?
		bool_matrix[:,1] = (((new_glomeruli[:,3]+r_glom).__ge__(i[3]-GoCaxon_y/2.)) & ((new_glomeruli[:,3]-r_glom).__le__(i[3]+GoCaxon_y/2.)))
		# glomerulus falls into the x range of values?
		bool_matrix[:,2] = (((new_glomeruli[:,4]+r_glom).__ge__(i[4]-GoCaxon_z/2.)) & ((new_glomeruli[:,4]-r_glom).__le__(i[4]+GoCaxon_z/2.)))

		good_gloms = np.where(np.sum(bool_matrix, axis=1)==3)[0]	# finds indexes of glomeruli that, on the selected axis, have the correct sum value
		chosen_rand = np.random.permutation(good_gloms)
		good_gloms_matrix = new_glomeruli[chosen_rand]
		prob = np.sort((np.sqrt((good_gloms_matrix[:,2]-i[2])**2 + (good_gloms_matrix[:,3]-i[3])**2))/150.)

		b=np.zeros(len(good_gloms_matrix))

		for idx1,j in enumerate(good_gloms_matrix):

			if idx <= n_conn_goc:

				ra = np.random.random()
				if (ra).__gt__(prob[idx1]):

					idx += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = i[0]
					matrix[0,1] = j[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					axonGOC_glom = np.vstack((axonGOC_glom, matrix))

					new_glomeruli[int(j[0] )- first_glomerulus,:] = OoB_value

	axonGOC_glom = axonGOC_glom[1:-1,:]

	return axonGOC_glom



def connectome_goc_grc(glom_grc, goc_glom):
	# Given goc --> glom connectivity, infer goc --> granules
	# First, create a DataFrame where column names are glomeruli and values are granules
	# contacted by each glomerulus
	glom_grc_df = pd.DataFrame()
	for key in np.unique(glom_grc[:,0]): # for each glomerulus connected to one or more granule cell
		target_grcs = glom_grc[glom_grc[:,0]==key,1]

	# Now store in a dictionary all granules inhibited by each Golgi cell:
	# keys --> Golgi cells
	# values --> contacted Granule cells
	goc_grc_dic = {}
	for key in np.unique(goc_glom[:,0]):
		target_glom = np.array(goc_glom[goc_glom[:,0]==key,1], dtype=int)
		# Filter out any Glomerulus not connected to any granular cells.
		target_glom = list(filter(lambda x: x in glom_grc_df.index, target_glom))
		target_grc = np.unique(glom_grc_df[target_glom])
		target_grc = target_grc[~np.isnan(target_grc)]
		goc_grc_dic[key] = target_grc

	# Finally, turn everything into a matrix
	pres = np.concatenate([np.tile(key, len(val)) for key, val in goc_grc_dic.items()])
	post = np.concatenate([val for val in goc_grc_dic.values()])
	goc_grc = np.column_stack([pres, post])

	return goc_grc



# Connectivity between ascending axon of GrCs and GoCs & connectivity between parallel fibers and GoCs
def connectome_grc_goc(first_granule, granules, golgicells, r_goc_vol, OoB_value, n_connAA, n_conn_pf, tot_conn, aa_goc, pf_goc):

	densityWarningSent = False
	new_granules = np.copy(granules)
	new_golgicells = np.random.permutation(golgicells)
	if new_granules.shape[0] <= new_golgicells.shape[0]:
		raise Exception("The number of golgi cells was less than the number of granule cells. Simulation cannot continue.")


	# for all Golgi cells: calculate which ascending axons of GrCs fall into the area of GoC soma, then choose 400 of them for the connection and delete them from successive computations, since 1 axon is connected to 1 GoC
	for i in new_golgicells:

		idx = 1
		connectedAA = np.array([])

		axon_matrix = (((new_granules[:,2]-i[2])**2)+((new_granules[:,4]-i[4])**2)-(r_goc_vol**2)).__le__(0)
		goodAA = np.where(axon_matrix==True)[0]		# finds indexes of ascending axons that can potentially be connected
		chosen_rand = np.random.permutation(goodAA)
		goodAA_matrix = new_granules[chosen_rand]
		prob = np.sort((np.sqrt((goodAA_matrix[:,2]-i[2])**2 + (goodAA_matrix[:,4]-i[4])**2))/r_goc_vol)

		for ind,j in enumerate(goodAA_matrix):
			if idx <= n_connAA:
				ra = np.random.random()
				if (ra).__gt__(prob[ind]):
					idx += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = j[0]
					matrix[0,1] = i[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					aa_goc = np.vstack((aa_goc, matrix))
					connectedAA = np.append(connectedAA,j[0])

		# parallel fiber and GoC connections

		good_grc = np.delete(granules, (connectedAA - first_granule), 0)
		intersections = (good_grc[:,2]).__ge__(i[2]-r_goc_vol) & (good_grc[:,2]).__le__(i[2]+r_goc_vol)
		good_pf = np.where(intersections==True)[0]				# finds indexes of granules that can potentially be connected

		# The remaining amount of parallel fibres to connect after subtracting the amount of already connected ascending axons.
		parallelFibersToConnect = tot_conn - len(connectedAA)

		# Randomly select parallel fibers to be connected with a GoC, to a maximum of tot_conn connections
		# TODO: Calculate the risk of not having enough granule cells beforehand, outside of the for loop for performance.
		if good_pf.shape[0] < parallelFibersToConnect:
			connected_pf = np.random.choice(good_pf, min(tot_conn-len(connectedAA), good_pf.shape[0]), replace = False)
			totalConnectionsMade = connected_pf.shape[0] + len(connectedAA)
			# Warn the user once if not enough granule cells are present to connect to the Golgi cell.
			if not densityWarningSent:
				densityWarningSent = True
				print("[WARNING] The granule cell density is too low compared to the Golgi cell density to make physiological connections!")
		else:
			connected_pf = np.random.choice(good_pf, tot_conn-len(connectedAA), replace = False)
			totalConnectionsMade = tot_conn

		pf_idx = good_grc[connected_pf,:]

		matrix_pf = np.zeros((totalConnectionsMade, 3))	# construction of the output matrix
		matrix_pf[:,1] = i[0]
		matrix_pf[0:len(connectedAA),0] = connectedAA
		matrix_pf[len(connectedAA):totalConnectionsMade,0] = pf_idx[:,0]

		# Store Euclidean distance.
		matrix_pf[0:len(connectedAA),2] = np.sqrt((granules[(connectedAA.astype(int)-first_granule),2]-i[2])**2 + (granules[(connectedAA.astype(int)-first_granule),3]-i[3])**2 + (granules[(connectedAA.astype(int)-first_granule),4]-i[4])**2)
		matrix_pf[len(connectedAA):totalConnectionsMade,2] = np.sqrt((pf_idx[:,2]-i[2])**2 + (pf_idx[:,3]-i[3])**2 + (pf_idx[:,4]-i[4])**2)
		pf_goc = np.vstack((pf_goc, matrix_pf))

		new_granules[((connectedAA.astype(int)) - first_granule),:] = OoB_value

		# End of Golgi cell loop

	aa_goc = aa_goc[1:-1,:]
	pf_goc = pf_goc[1:-1,:]

	aa_goc = aa_goc[aa_goc[:,1].argsort()]
	pf_goc = pf_goc[pf_goc[:,1].argsort()]		# sorting of the resulting vector on the post-synaptic neurons

	return aa_goc, pf_goc



# GRANULAR-MOLECULAR LAYERS

# Connectivity between ascending axons and PC dendrites
def connectome_aa_pc(first_granule, granules, purkinjes, x_pc, z_pc, OoB_value, aa_pc):

	new_granules = np.copy(granules)

	# for all Purkinje cells: calculate and choose which granules fall into the area of PC dendritic tree, then delete them from successive computations, since 1 ascending axon is connected to only 1 PC
	for i in purkinjes:

		# CONTROLLARE X E Z
		bool_matrix = np.zeros((new_granules.shape[0], 2))	# matrix initialization
		# ascending axon falls into the z range of values?
		bool_matrix[:,0] = (new_granules[:,4]).__ge__(i[4]-z_pc/2.) & (new_granules[:,4]).__le__(i[4]+z_pc/2.)
		# ascending axon falls into the x range of values?
		bool_matrix[:,1] = (new_granules[:,2]).__ge__(i[2]-x_pc/2.) & (new_granules[:,2]).__le__(i[2]+x_pc/2.)

		good_aa = np.where(np.sum(bool_matrix, axis=1)==2)[0]	# finds indexes of ascending axons that, on the selected axis, have the correct sum value

		# construction of the output matrix: the first column has the GrC id, while the second column has the PC id
		matrix = np.zeros((len(good_aa), 3))
		matrix[:,1] = i[0]
		matrix[:,0] = good_aa + first_granule
		matrix[:,2] = np.sqrt((granules[good_aa,2]-i[2])**2 + (granules[good_aa,3]-i[3])**2 + (granules[good_aa,4]-i[4])**2)
		aa_pc = np.vstack((aa_pc, matrix))

		new_granules[good_aa,:] = OoB_value	# update the granules matrix used for computation by deleting the coordinates of connected ones

	aa_pc = aa_pc[1:-1,:]

	return aa_pc



# Connectivity between parallel fibers and PC dendrites
def connectome_pf_pc(first_granule, granules, purkinjes, x_pc, pf_pc):

	# for all Purkinje cells: calculate and choose which parallel fibers fall into the area of PC dendritic tree (then delete them from successive computations, since 1 parallel fiber is connected to a maximum of PCs)
	for i in purkinjes:

		# which parallel fibers fall into the x range of values?
		bool_matrix = (granules[:,2]).__ge__(i[2]-x_pc/2.) & (granules[:,2]).__le__(i[2]+x_pc/2.)	# CAMBIARE IN new_granules SE VINCOLO SU 30 pfs
		good_pf = np.where(bool_matrix==True)[0]	# finds indexes of parallel fibers that, on the selected axis, satisfy the condition

		# construction of the output matrix: the first column has the GrC id, while the second column has the PC id
		matrix = np.zeros((len(good_pf), 3))
		matrix[:,1] = i[0]
		matrix[:,0] = good_pf + first_granule
		matrix[:,2] = np.sqrt((granules[good_pf,2]-i[2])**2 + (granules[good_pf,3]-i[3])**2 + (granules[good_pf,4]-i[4])**2)
		pf_pc = np.vstack((pf_pc, matrix))

	pf_pc = pf_pc[1:-1,:]

	return pf_pc



# Connections between basket cells and parallel fibers
def connectome_pf_bc(first_granule, basketcells, granules, r_sb, h_pf, pf_bc):

	for i in basketcells:	# for each basket cell find all the parallel fibers that fall into the sphere with centre the cell soma and appropriate radius

		# find all cells that satisfy the condition
		bc_matrix = (((granules[:,2]-i[2])**2)+((h_pf-i[3])**2)-(r_sb**2)).__le__(0)
		good_pf = np.where(bc_matrix==True)[0]	# indexes of basket cells that can potentially be connected

		matrix = np.zeros((len(good_pf), 3))
		matrix[:,1] = i[0]
		matrix[:,0] = good_pf + first_granule
		matrix[:,2] = np.sqrt((granules[good_pf,2]-i[2])**2 + (granules[good_pf,3]-i[3])**2 + (granules[good_pf,4]-i[4])**2)
		pf_bc = np.vstack((pf_bc, matrix))

	pf_bc = pf_bc[1:-1,:]

	return pf_bc



# Connections between basket cells and parallel fibers
def connectome_pf_sc(first_granule, stellates, granules, r_sb, h_pf, pf_sc):

	for i in stellates:	# for each stellate cell find all the parallel fibers that fall into the sphere with centre the cell soma and appropriate radius

		# find all cells that satisfy the condition
		sc_matrix = (((granules[:,2]-i[2])**2)+((h_pf-i[3])**2)-(r_sb**2)).__le__(0)
		good_pf = np.where(sc_matrix==True)[0]	# indexes of stellate cells that can potentially be connected

		matrix = np.zeros((len(good_pf), 3))
		matrix[:,1] = i[0]
		matrix[:,0] = good_pf + first_granule
		matrix[:,2] = np.sqrt((granules[good_pf,2]-i[2])**2 + (granules[good_pf,3]-i[3])**2 + (granules[good_pf,4]-i[4])**2)
		pf_sc = np.vstack((pf_sc, matrix))

	pf_sc = pf_sc[1:-1,:]

	return pf_sc



# MOLECULAR LAYER

# Connectivity between basket and stellate cells and PCs
def connectome_sc_bc_pc(first_stellate, first_basket, basketcells, stellates, purkinjes, distx, distz, conv, sc_pc, bc_pc):

	for i in purkinjes:	# for all Purkinje cells: calculate which basket and stellate cells can be connected, then choose 20 of them for each typology

		idx_bc = 1
		idx_sc = 1

		# find all cells that satisfy the distance condition for both types
		sc_matrix = (np.absolute(stellates[:,4]-i[4])).__lt__(distz) & (np.absolute(stellates[:,2]-i[2])).__lt__(distx)
		#bc_matrix = (np.absolute(basketcells[:,4]-i[4])).__lt__(distz) & (np.absolute(basketcells[:,2]-i[2])).__lt__(distx)
		bc_matrix = (np.absolute(basketcells[:,4]-i[4])).__lt__(distx) & (np.absolute(basketcells[:,2]-i[2])).__lt__(distz)


		good_bc = np.where(bc_matrix==True)[0]	# indexes of basket cells that can potentially be connected
		good_sc = np.where(sc_matrix==True)[0]	# indexes of stellate cells that can potentially be connected

		chosen_rand_bc = np.random.permutation(good_bc)
		good_bc_matrix = basketcells[chosen_rand_bc]
		chosen_rand_sc = np.random.permutation(good_sc)
		good_sc_matrix = stellates[chosen_rand_sc]


		# basket cells connectivity
		for j in good_bc_matrix:

			if idx_bc <= conv:

				ra = np.random.random()
				if (ra).__gt__((np.absolute(j[4]-i[4]))/distx) & (ra).__gt__((np.absolute(j[2]-i[2]))/distz):

					idx_bc += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = j[0]
					matrix[0,1] = i[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					bc_pc = np.vstack((bc_pc, matrix))


		# stellate cells connectivity
		for k in good_sc_matrix:

			if idx_sc <= conv:

				ra = np.random.random()
				if (ra).__gt__((np.absolute(k[4]-i[4]))/distz) & (ra).__gt__((np.absolute(k[2]-i[2]))/distx):

					idx_sc += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = k[0]
					matrix[0,1] = i[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					sc_pc = np.vstack((sc_pc, matrix))

	sc_pc = sc_pc[1:-1,:]
	bc_pc = bc_pc[1:-1,:]

	return sc_pc, bc_pc


# gap junction connectivity algorithm for basket cells
def gap_junctions_bc(first_basket, basketcells, d_xy, d_z, dc_gj, gj_bc):

	for i in basketcells:	# for each basket cell calculate the distance with every other cell of the same type in the volume, than choose 4 of them

		idx = 1

		# find all cells that satisfy the distance condition
		bc_matrix = (np.absolute(basketcells[:,4]-i[4])).__lt__(d_z) & (np.absolute(basketcells[:,4]-i[4])).__ne__(0) & (np.sqrt((basketcells[:,2]-i[2])**2 + (basketcells[:,3]-i[3])**2)).__lt__(d_xy)
		good_bc = np.where(bc_matrix==True)[0]	# indexes of basket cells that can potentially be connected
		chosen_rand = np.random.permutation(good_bc)
		good_bc_matrix = basketcells[chosen_rand]

		for j in good_bc_matrix:

			if idx <= dc_gj:

				ra = np.random.random()
				if (ra).__gt__((np.absolute(j[4]-i[4]))/float(d_z)) & (ra).__gt__((np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2))/float(d_xy)):

					idx += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = i[0]
					matrix[0,1] = j[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					gj_bc = np.vstack((gj_bc, matrix))

	gj_bc = gj_bc[1:-1,:]

	return gj_bc


# gap junction connectivity algorithm for basket cells
def gap_junctions_sc(first_stallate, stellates, d_xy, d_z, dc_gj, gj_sc):

	for i in stellates:	# for each stellate cell calculate the distance with every other cell of the same type in the volume, then choose 4 of them

		idx = 1

		# find all cells that satisfy the distance condition
		sc_matrix = (np.absolute(stellates[:,4]-i[4])).__lt__(d_z) & (np.absolute(stellates[:,4]-i[4])).__ne__(0) & (np.sqrt((stellates[:,2]-i[2])**2 + (stellates[:,3]-i[3])**2)).__lt__(d_xy)
		good_sc = np.where(sc_matrix==True)[0]	# indexes of stellate cells that can potentially be connected
		chosen_rand = np.random.permutation(good_sc)
		good_sc_matrix = stellates[chosen_rand]

		for j in good_sc_matrix:

			if idx <= dc_gj:

				ra = np.random.random()
				if (ra).__gt__((np.absolute(j[4]-i[4]))/float(d_z)) & (ra).__gt__((np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2))/float(d_xy)):

					idx += 1

					matrix = np.zeros((1, 3))
					matrix[0,0] = i[0]
					matrix[0,1] = j[0]
					matrix[0,2] = np.sqrt((j[2]-i[2])**2 + (j[3]-i[3])**2 + (j[4]-i[4])**2)
					gj_sc = np.vstack((gj_sc, matrix))

	gj_sc = gj_sc[1:-1,:]

	return gj_sc

# Connectivity between Purkinje axons and glutamatergic neurons

# Since the dendritic tree of DCN glutamatergic cells has a lenght in the order of millimeters (range [341-4974] with mean of 2654+-1360 microm), here we consider it as a plane with a random orientation (that remains fixed for each single cell). We need a much larger simulation volume to consider its shape more accurately.

def connectome_pc_dcn(first_dcn, dcn_idx, purkinjes, dcn_glut, div_pc, dend_tree_coeff, pc_dcn):

	for i in purkinjes:	# for all Purkinje cells: calculate the distance with the area around glutamatergic DCN cells soma, then choose 4-5 of them

		distance = np.zeros((dcn_glut.shape[0]))
		distance = (np.absolute((dend_tree_coeff[:,0]*i[2])+(dend_tree_coeff[:,1]*i[3])+(dend_tree_coeff[:,2]*i[4])+dend_tree_coeff[:,3]))/(np.sqrt((dend_tree_coeff[:,0]**2)+(dend_tree_coeff[:,1]**2)+(dend_tree_coeff[:,2]**2)))

		dist_matrix = np.zeros((dcn_glut.shape[0],2))
		dist_matrix[:,1] = dcn_glut[:,0]
		dist_matrix[:,0] = distance
		dcn_dist = np.random.permutation(dist_matrix)

		# If the number of DCN cells are less than the divergence value, all neurons are connected to the corresponding PC
		if dcn_idx.shape[0]<div_pc:
			matrix = np.zeros((dcn_idx.shape[0], 3))
			matrix[:,0] = i[0]
			matrix[:,1] = dcn_idx
			matrix[:,2] = np.sqrt((dcn_glut[:,2]-i[2])**2 + (dcn_glut[:,3]-i[3])**2 + (dcn_glut[:,4]-i[4])**2)
			pc_dcn = np.vstack((pc_dcn, matrix))

		else:
			if np.random.rand()>0.5:

				connected_f = dcn_dist[0:div_pc,1]
				connected_dist = dcn_dist[0:div_pc,0]
				connected_provv = connected_f.astype(int)
				connected_dcn = connected_provv

				# construction of the output matrix: the first column has the  PC index, while the second column has the connected DCN cell index
				matrix = np.zeros((div_pc, 3))
				matrix[:,0] = i[0]
				matrix[:,1] = connected_dcn
				matrix[:,2] = np.sqrt((dcn_glut[(connected_dcn-first_dcn),2]-i[2])**2 + (dcn_glut[(connected_dcn-first_dcn),3]-i[3])**2 + (dcn_glut[(connected_dcn-first_dcn),4]-i[4])**2)
				pc_dcn = np.vstack((pc_dcn, matrix))

			else:

				connected_f = dcn_dist[0:(div_pc-1),1]
				connected_dist = dcn_dist[0:(div_pc-1),0]
				connected_provv = connected_f.astype(int)
				connected_dcn = connected_provv

				matrix = np.zeros(((div_pc-1), 3))
				matrix[:,0] = i[0]
				matrix[:,1] = connected_dcn
				matrix[:,2] = np.sqrt((dcn_glut[(connected_dcn-first_dcn),2]-i[2])**2 + (dcn_glut[(connected_dcn-first_dcn),3]-i[3])**2 + (dcn_glut[(connected_dcn-first_dcn),4]-i[4])**2)
				pc_dcn = np.vstack((pc_dcn, matrix))

	pc_dcn = pc_dcn[1:-1,:]

	return pc_dcn

def connectome_glom_dcn(first_glomerulus, glomeruli, dcn_glut, conv_dcn, glom_dcn):
	for i in dcn_glut:

		connected_gloms = np.random.choice(glomeruli[:,0], conv_dcn, replace=False)

		matrix = np.zeros((conv_dcn, 3))
		matrix[:,0] = connected_gloms.astype(int)
		matrix[:,1] = i[0]
		matrix[:,2] = np.sqrt((glomeruli[(connected_gloms.astype(int)-first_glomerulus),2]-i[2])**2 + (glomeruli[(connected_gloms.astype(int)-first_glomerulus),3]-i[3])**2 + (glomeruli[(connected_gloms.astype(int)-first_glomerulus),4]-i[4])**2)

		glom_dcn = np.vstack((glom_dcn, matrix))

	glom_dcn = glom_dcn[1:-1,:]
	return glom_dcn


def connectome_gj_goc(r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgicells):
	gj_goc = np.zeros((1,3))
	for i in golgicells:	# for each Golgi find all cells of the same type that, through their dendritic tree, fall into its axonal tree

		# do not consider the current cell in the connectivity calculus
		a = np.where(golgicells[:,0]==i[0])[0]
		del_goc = np.delete(golgicells[:,0], a)
		potential_goc = del_goc.astype(int)

		bool_matrix = np.zeros(((golgicells.shape[0])-1, 3))
		bool_matrix[:,0] = (np.absolute(golgicells[potential_goc,2]-i[2])).__le__(r_goc_vol + (GoCaxon_x/2.))
		bool_matrix[:,1] = (np.absolute(golgicells[potential_goc,3]-i[3])).__le__(r_goc_vol + (GoCaxon_y/2.))
		bool_matrix[:,2] = (np.absolute(golgicells[potential_goc,4]-i[4])).__le__(r_goc_vol + (GoCaxon_z/2.))

		good_goc = np.where(np.sum(bool_matrix, axis=1)==3)[0]	# finds indexes of Golgi cells that satisfy all conditions

		matrix = np.zeros((len(good_goc), 3))
		matrix[:,0] = i[0]
		matrix[:,1] = good_goc
		matrix[:,2] = np.sqrt((golgicells[good_goc,2]-i[2])**2 + (golgicells[good_goc,3]-i[3])**2 + (golgicells[good_goc,4]-i[4])**2)
		#matrix[:,2] = np.sqrt((golgicells[good_goc.astype(int),2]-i[2])**2 + (golgicells[good_goc.astype(int),3]-i[3])**2 + (golgicells[good_goc.astype(int),4]-i[4])**2)
		gj_goc = np.vstack((gj_goc, matrix))

	gj_goc = gj_goc[1:-1,:]
	return gj_goc
