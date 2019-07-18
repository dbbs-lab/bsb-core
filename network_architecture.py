import h5py
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import distance
from scaffold_params import *
from scaffold_functions import *
from pprint import pprint

goc_in_volume, goc_eps, goc_sublayers, goc_height_placement, ngoc_per_sublayer, goc_bounds = sublayer_partitioning('granular', 'golgi', volume_base_size)
placement_stats['golgi']['total_n_golgi'] = goc_in_volume
placement_stats['golgi']['golgi_subl'] = ngoc_per_sublayer

v_occupied_goc = volume_occupied(goc_in_volume, cells_radius['golgi'])
glom_in_volume, glom_eps, glom_sublayers, glom_height_placement, nglom_per_sublayer, glom_bounds = sublayer_partitioning('granular', 'glomerulus', volume_base_size, v_occupied_goc)
placement_stats['glomerulus']['total_n_glomerulus'] = 	glom_in_volume
placement_stats['glomerulus']['glomerulus_subl'] = nglom_per_sublayer

v_occupied_glom = volume_occupied(glom_in_volume, cells_radius['glomerulus'])
grc_in_volume, grc_eps, grc_sublayers, grc_height_placement, ngrc_per_sublayer, grc_bounds = sublayer_partitioning('granular', 'granule', volume_base_size, v_occupied_goc, v_occupied_glom)
placement_stats['granule']['total_n_granule'] = 	grc_in_volume
placement_stats['granule']['granule_subl'] = ngrc_per_sublayer

sc_in_volume, sc_eps, sc_sublayers, sc_height_placement, nsc_per_sublayer, sc_bounds = sublayer_partitioning('molecular', 'stellate', volume_base_size)
bc_in_volume, bc_eps, bc_sublayers, bc_height_placement, nbc_per_sublayer, bc_bounds = sublayer_partitioning('molecular', 'basket', volume_base_size)
placement_stats['stellate']['total_n_stellate'] = sc_in_volume
placement_stats['stellate']['stellate_subl'] = nsc_per_sublayer
placement_stats['basket']['total_n_basket'] = bc_in_volume
placement_stats['basket']['basket_subl'] = nbc_per_sublayer

dcn_in_volume, dcn_eps, dcn_sublayers, dcn_height_placement, ndcn_per_sublayer, dcn_bounds = sublayer_partitioning('dcn', 'dcn', dcn_volume)
dcn_eps /= 10.   # Reducing the average dcn_eps - this number still hard-coded :/
placement_stats['dcn']['total_n_dcn'] = dcn_in_volume
placement_stats['dcn']['dcn_subl'] = ndcn_per_sublayer




if goc_in_volume > 0.0:
    cells_placement(goc_sublayers, volume_base_size, 'golgi', goc_eps, goc_height_placement, ngoc_per_sublayer, goc_bounds)
if glom_in_volume > 0.0:
    cells_placement(glom_sublayers, volume_base_size,'glomerulus', glom_eps, glom_height_placement, nglom_per_sublayer, glom_bounds)
if grc_in_volume > 0.0:
    cells_placement(grc_sublayers, volume_base_size,'granule', grc_eps, grc_height_placement, ngrc_per_sublayer, grc_bounds)
if pc_in_volume > 0.0:
    purkinje_placement(pc_extension_dend_tree, pc_in_volume)
if sc_in_volume > 0.0:
    cells_placement(sc_sublayers, volume_base_size,'stellate', sc_eps, sc_height_placement, nsc_per_sublayer, sc_bounds)
if bc_in_volume > 0.0:
    cells_placement(bc_sublayers, volume_base_size,'basket', bc_eps, bc_height_placement, nbc_per_sublayer, bc_bounds)
if dcn_in_volume > 0.0:
    cells_placement(dcn_sublayers, dcn_volume, 'dcn', dcn_eps, dcn_height_placement, ndcn_per_sublayer, dcn_bounds)

### TEMP: add a field to final_cell_positions for IO ###
### We randomly place 7 IO (for a 400.0 x 400.0 scaffold with 69 Purkinje cells) in a 400 x 400 x 100 volume
# io_pos = np.random.random((7,3)) * np.array([400, 100, 400])
# final_cell_positions['IO'] = io_pos
#########################

adapt_positions()

#### TEMP for IO positions ####
# for key, pos in final_cell_positions.iteritems():
	# if key != 'IO':
		# pos[:,1] += 100
###############################

######## Prepare data for placement saving in .hdf5 format #######
# Progressive indexing of neurons
# 2.7 ==> 3.7: .itervalues() deprecated in favor of .values()
cel_num_vec = sorted(cell_type_ID.values())

# 2.7 ==> 3.7: .iteritems() deprecated in favor of .items()
cellID2type = {val: key for key, val in cell_type_ID.items()}
#### TEMP ####
# cel_num_vec.append(8)
# cellID2type[8]= 'IO'
# cell_type_ID['IO'] = 8
# y_io = 100
##############
max_shape_values = [np.array(final_cell_positions[cellID2type[val]]).shape[0] for val in cel_num_vec]
max_shape_values.insert(0,0)
prog_nums = []
for n in range(1,len(max_shape_values)):
	summed_prec = sum(max_shape_values[0:n])
	prog_nums.extend(range(summed_prec, summed_prec+max_shape_values[n]))

data_matrix = np.empty((1, 4))

for n in cel_num_vec:
	if len(final_cell_positions[cellID2type[n]]) == 0:
		continue
	cell_id = np.zeros(final_cell_positions[cellID2type[n]].shape[0])+cell_type_ID[cellID2type[n]]
	positions = final_cell_positions[cellID2type[n]]
	pos_id = np.column_stack((cell_id, positions))
	data_matrix = np.concatenate((data_matrix, pos_id))

data_matrix = data_matrix[1::]



#####################################################################################################################
############ Part2 - Building Connectivity

positions = np.column_stack((prog_nums, data_matrix))

# simulation volume
y_ml = layers_thick['molecular']		# height of molecular layer
y_pc = layers_thick['purkinje']	# height of PC soma layer
y_gl = layers_thick['granular']		# height of granular layer
y_dcn = layers_thick['dcn']		# height of DCN layer

OoB_value = 100000.	# Out of Bounds number, when an already selected cell must not be taken for further consideration

# Glomeruli parameters
r_glom = cells_radius['glomerulus']	# radius of glomerulus soma

x_pc = pc_extension_dend_tree	# Purkinje cell extention tree



# initialization of the output matrices

# molecular layer
sc_pc = np.zeros((1,3))
bc_pc = np.zeros((1,3))
gj_bc = np.zeros((1,3))
gj_sc = np.zeros((1,3))

# granular-molecular layer
aa_pc = np.zeros((1,3))
pf_pc = np.zeros((1,3))
pf_sc = np.zeros((1,3))
pf_bc = np.zeros((1,3))

# granular layer
glom_grc = np.zeros((1,3))
glom_bd = np.zeros((1,3))
axonGOC_glom = np.zeros((1,3))
aa_goc = np.zeros((1,3))
pf_goc = np.zeros((1,3))

# deep cerebellar nucleus
pc_dcn = np.zeros((1,3))

# Determine all the submatrices for the different cell types, with: cell index, type, xyz coordinates

golgi_idx = np.where(positions[:,1]==cell_type_ID['golgi'])[0]			# find all indexes where the cell type is that of the Golgi cells
if len(golgi_idx) == 0:
	golgicells = np.empty((0, positions.shape[1]))
else:
	first_golgi = golgi_idx[0]					# index of the first element
	last_golgi = golgi_idx[-1]					# index of the last element
	golgicells = positions[first_golgi:last_golgi+1,:]		# submatrix for Golgi cells

glomeruli_idx = np.where(positions[:,1]==cell_type_ID['glomerulus'])[0]			# find all indexes where the cell type is that of the glomeruli
if len(glomeruli_idx) == 0:
	glomeruli = np.empty((0, positions.shape[1]))
else:
	first_glomerulus = glomeruli_idx[0]
	last_glomerulus = glomeruli_idx[-1]
	glomeruli = positions[first_glomerulus:last_glomerulus+1,:]	# submatrix for glomeruli

granules_idx = np.where(positions[:,1]==cell_type_ID['granule'])[0]			# find all indexes where the cell type is that of the granules
if len(granules_idx) == 0:
	granules = np.empty((0, positions.shape[1]))
else:
	first_granule = granules_idx[0]
	last_granule = granules_idx[-1]
	granules = positions[first_granule:last_granule+1,:]		# submatrix for granules

purkinjes_idx = np.where(positions[:,1]==cell_type_ID['purkinje'])[0]			# find all indexes where the cell type is that of the Purkinje cells
if len(purkinjes_idx) == 0:
	purkinjes = np.empty((0, positions.shape[1]))
else:
	first_pc = purkinjes_idx[0]
	last_pc = purkinjes_idx[-1]
	purkinjes = positions[first_pc:last_pc+1,:]			# submatrix for Purkinje cells

basket_idx = np.where(positions[:,1]==cell_type_ID['basket'])[0]			# find all indexes where the cell type is that of the basket cells
if len(basket_idx) == 0:
	basketcells = np.empty((0, positions.shape[1]))
else:
	first_basket = basket_idx[0]
	last_basket = basket_idx[-1]
	basketcells = positions[first_basket:last_basket+1,:]		# submatrix for basket cells

stellates_idx = np.where(positions[:,1]==cell_type_ID['stellate'])[0]			# find all indexes where the cell type is that of the stellate cells
if len(stellates_idx) == 0:
	stellates = np.empty((0, positions.shape[1]))
else:
	first_stellate = stellates_idx[0]
	last_stellate = stellates_idx[-1]
	stellates = positions[first_stellate:last_stellate+1,:]		# submatrix for stellate cells

dcn_idx = np.where(positions[:,1]==cell_type_ID['dcn'])[0]			# find all indexes where the cell type is that of the dcn glutamatergic cells
if len(dcn_idx) == 0:
	dcn_glut = np.empty((0, positions.shape[1]))
else:
	first_dcn = dcn_idx[0]
	last_dcn = dcn_idx[-1]
	dcn_glut = positions[first_dcn:last_dcn+1,:]			# submatrix for dcn glutamatergic cells

# Calculate the heights of all parallel fibers of the simulation volume
h_pf = np.zeros((len(granules_idx),2))

for idx,i in enumerate(granules):

	h_final = 0
	h_max = h_m + 30. + sd + i[3]
	h_min = h_m + 30. - sd + i[3]

	if (h_max <= (y_dcn + y_gl + y_pc + y_ml)) & (h_min > (y_dcn + y_gl + y_pc)):
		h_range = np.arange(h_min, h_max, 0.01)
		h_final = np.random.choice(h_range)
	#	h_final = np.random.normal((h_m + i[3]), sd/4.0)
	#	h_pf[idx,0] = i[0]
	#	h_pf[idx,1] = np.random.normal((h_m + i[3]), sd/4.0)	# to keep the value into the boundaries of both the physiological pf height and those of the molecular layer

	elif (h_max <= (y_dcn + y_gl + y_pc + y_ml)) & (h_min <= (y_dcn + y_gl + y_pc)):
	#	sd1 = (h_max - (y_dcn + y_gl + y_pc))/2.
	#	h = h_max - sd1
		h_range = np.arange(y_dcn + y_gl + y_pc, h_max, 0.01)
	#	h_range = np.arange(y_dcn + y_gl + y_pc - 10., h_max, 0.01)
		h_final = np.random.choice(h_range)
	#	h_final = np.random.normal(h, sd1/4.0)
	#	h_pf[idx,0] = i[0]
	#	h_pf[idx,1] = np.random.normal(h, sd1/4.0)

	else:
	#	sd1 = ((y_dcn + y_gl + y_pc + y_ml) - h_min)/2.
	#	h = (y_dcn + y_gl + y_pc + y_ml) - sd1
		h_range = np.arange(h_min, y_dcn + y_gl + y_pc + y_ml, 0.01)
	#	h_range = np.arange(h_min, y_dcn + y_gl + y_pc + y_ml + 10., 0.01)
		h_final = np.random.choice(h_range)
	#	h_final = np.random.normal(h, sd1/4.0)
	#	h_pf[idx,0] = i[0]
	#	h_pf[idx,1] = np.random.normal(h, sd1/4.0)

	h_pf[idx,0] = i[0]
	h_pf[idx,1] = h_final

	# check that the height falls into the molecular layer
	if h_pf[idx,1] < (y_dcn + y_gl + y_pc):
		h_final = y_dcn + y_gl + y_pc + 1#np.random.choice(np.arange(y_dcn + y_gl + y_pc, y_dcn + y_gl + y_pc + 10., 0.01))
	#	h_pf[idx,0] = i[0]
	#	h_pf[idx,1] = y_dcn + y_gl + y_pc + 1
	elif h_pf[idx,1] >= (y_dcn + y_gl + y_pc + y_ml):
		h_final = y_dcn + y_gl + y_pc + y_ml - 1#np.random.choice(np.arange(y_dcn + y_gl + y_pc + y_ml - 10., y_dcn + y_gl + y_pc + y_ml, 0.01))
	#	h_pf[idx,0] = i[0]
	#	h_pf[idx,1] = y_dcn + y_gl + y_pc + y_ml - 1

	h_pf[idx,0] = i[0]
	h_pf[idx,1] = h_final

#h_pf_mat = np.column_stack((granules[:,0], h_pf))
h_pf_mat = h_pf



# connections between glomeruli and granules dendrites
glom_grc = connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom, glom_grc)
# connections between glomeruli and Golgi cells basolateral dendrites
glom_goc = connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol, glom_bd)
# connections between, respectively, granules ascending axons and parallel fibers and Golgi cells dendrites
aa_goc, pf_goc  = connectome_grc_goc(first_granule, granules, golgicells, r_goc_vol, OoB_value, n_connAA, n_conn_pf, tot_conn, aa_goc, pf_goc)
# connections between Golgi cells axon and glomeruli
goc_glom = connectome_goc_glom(first_glomerulus, glomeruli, golgicells, GoCaxon_x, GoCaxon_y, GoCaxon_z, r_glom, n_conn_goc, OoB_value, axonGOC_glom)
# connections between Golgi cells axon and granule cells dendrite
goc_grc = connectome_goc_grc(glom_grc, goc_glom)
# connections between granules ascending axons and Purkinje cells dendritic tree
aa_pc = connectome_aa_pc(first_granule, granules, purkinjes, x_pc, z_pc, OoB_value, aa_pc)
# connections between parallel fibers and Purkinje cells dendritic tree
pf_pc = connectome_pf_pc(first_granule, granules, purkinjes, x_pc, pf_pc)
# connections between parallel fibers and basket cells dendritic tree
pf_bc = connectome_pf_bc(first_granule, basketcells, granules, r_sb, h_pf[:,1], pf_bc)
# connections between parallel fibers and stellate cells dendritic tree
pf_sc = connectome_pf_sc(first_granule, stellates, granules, r_sb, h_pf[:,1], pf_sc)
# connections between, respectively, stellate and basket cells and Purkinje dendritic tree
sc_pc, bc_pc = connectome_sc_bc_pc(first_stellate, first_basket, basketcells, stellates, purkinjes, distx, distz, conv, sc_pc, bc_pc)
# gap junctions of stellate cells
gj_sc = gap_junctions_sc(first_stellate, stellates, d_xy, d_z, dc_gj, gj_sc)
# gap junctions of basket cells
gj_bc = gap_junctions_bc(first_basket, basketcells, d_xy, d_z, dc_gj, gj_bc)
# golgi - golgi inhibition
gj_goc = connectome_gj_goc(r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgicells)



# Plane orientation for every DCN cell
dend_tree_coeff = np.zeros((dcn_glut.shape[0],4))

# PURKINJE-DCN LAYERS
for idx1 in enumerate(dcn_glut):

	if np.random.rand() > 0.5:
		dend_tree_coeff[idx1[0],0] = np.random.rand()#*dcn_volume[0]
	else:
		dend_tree_coeff[idx1[0],0] = (-1)*np.random.rand()#*dcn_volume[0]

	if np.random.rand() > 0.5:
		dend_tree_coeff[idx1[0],1] = np.random.rand()#*y_dcn
	else:
		dend_tree_coeff[idx1[0],1] = (-1)*np.random.rand()#*y_dcn

	if np.random.rand() > 0.5:
		dend_tree_coeff[idx1[0],2] = np.random.rand()#*dcn_volume[1]
	else:
		dend_tree_coeff[idx1[0],2] = (-1)*np.random.rand()#*dcn_volume[1]

	dend_tree_coeff[idx1[0],3] = (-1)*((dend_tree_coeff[idx1[0],0]*dcn_glut[idx1[0],2])+(dend_tree_coeff[idx1[0],1]*dcn_glut[idx1[0],3])+(dend_tree_coeff[idx1[0],2]*dcn_glut[idx1[0],4]))

if dcn_glut.shape[0] != 0:
	dcn_angle = np.column_stack((dcn_glut[:,0], dend_tree_coeff))

	# connectivity between Purkinje cells axon and glutamatergic DCN cells
	pc_dcn = connectome_pc_dcn(first_dcn, dcn_idx, purkinjes, dcn_glut, div_pc, dend_tree_coeff, pc_dcn)

	# initialization of the output matrix
	glom_dcn = np.zeros((1,3))

	glom_dcn = connectome_glom_dcn(first_glomerulus, glomeruli, dcn_glut, conv_dcn, glom_dcn)
else:
	dcn_angle = dend_tree_coeff
	pc_dcn = np.empty((0, pf_pc.shape[1]))
	glom_dcn = np.empty((0, pf_pc.shape[1]))


### TEMP connections from IO to PCs
# io_gid = positions[positions[:,1]==8,0]
# pcs_gid = positions[positions[:,1]==4,0]
# np.random.shuffle(pcs_gid)
# target_pcs_blocks = np.array_split(pcs_gid, 7)
# io_pc =np.concatenate([np.column_stack([np.repeat(val,len(target_pcs_blocks[idx])), target_pcs_blocks[idx]])
		# for idx, val in enumerate(io_gid)])
#####################################################################
f = h5py.File(save_name, 'w')
f.create_dataset('positions', data=positions)
f.create_dataset('hpf', data=h_pf_mat)
f.create_dataset('DCNangle', data=dcn_angle)
f.create_group('connections')
f['connections'].create_dataset('glom_grc', data=glom_grc)
f['connections'].create_dataset('glom_goc', data=glom_goc)
f['connections'].create_dataset('aa_goc', data=aa_goc)
f['connections'].create_dataset('pf_goc', data=pf_goc)
f['connections'].create_dataset('goc_glom', data=goc_glom)
f['connections'].create_dataset('goc_grc', data=goc_grc)
f['connections'].create_dataset('aa_pc', data=aa_pc)
f['connections'].create_dataset('pf_pc', data=pf_pc)
f['connections'].create_dataset('pf_bc', data=pf_bc)
f['connections'].create_dataset('pf_sc', data=pf_sc)
f['connections'].create_dataset('bc_pc', data=bc_pc)
f['connections'].create_dataset('sc_pc', data=sc_pc)
f['connections'].create_dataset('gj_bc', data=gj_bc)
f['connections'].create_dataset('gj_sc', data=gj_sc)
f['connections'].create_dataset('gj_goc', data=gj_goc)
f['connections'].create_dataset('pc_dcn', data=pc_dcn)
f['connections'].create_dataset('glom_dcn', data=glom_dcn)
#f['connections'].create_dataset('io_pc', data=io_pc)
f.close()
