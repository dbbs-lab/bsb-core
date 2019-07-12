import numpy as np

# Set the following variables to 1 to save / plot results
# Set to 0 otherwise
save = 1
plot = 1

########################################################################################
############################### PLACEMENT PARAMETERS ##################################
# The volume base_size can be changed without strict constraints;
# BTW, for single core simulations, size should be kept within 200 - 600 micron range
#
# N.B. by now, for semplicity, keep the two values equal
#volume_base_size = np.array([300., 300.])
base_size = 400.
volume_base_size = np.array([base_size, base_size])
dcn_volume = volume_base_size / 2

# Name of data file
filename = 'scaffold_full_IO'
save_name = '{}_{}x{}_v3'.format(filename, volume_base_size[0], volume_base_size[1])
save_name = save_name + '.hdf5'

# Purkinje / DCN ratio: the number of PC per DCN - (Note: describe better)
pc_dcn_ratio = 11.

# Extension of Purkinje cell dendritic tree
pc_extension_dend_tree = 130.
z_pc = 3.5	# NOME DA MODIFICARE - PC extension of the dendritic tree along the z-axis

# Thickness of different layers
layers_thick = {'granular': 150.,
				'purkinje': 30.,
				'molecular': 150.,
				'dcn': 600.}
# Soma radius of each cell type (micron)
''' Diameter of DCN Glutamatergic neuron is in range 15 - 35 micron (Aizemann et al., 2003)
	==> mean diam = 25 micron
	==> mean radius = 12.5 micron
	Slightly different estimate (Gauck and Jaeger, 2000): 10 - 35 micron, average = 20
	==> mean radius = 10 micron'''

cells_radius = {'golgi': 8,
				'glomerulus': 1.5,
				'granule': 2.5,
				'purkinje': 7.5,
				'stellate': 4.,
				'basket': 6.,
				'dcn': 10}


# Density distribution of each cell type
cells_density = {'golgi': 9*(10**(-6)),
				 'glomerulus': 3*(10**(-4)),
				 'granule': 3.9*(10**(-6)),
				 'purkinje': 0.45*(10**(-3)),
				 'stellate':1.0/2*10**(-4),
				 'basket':1.0/2*(10**(-4))}

# Cell type ID (can be changed without constraints)
cell_type_ID = {'golgi': 1,
				'glomerulus': 2,
				'granule': 3,
				'purkinje': 4,
				'basket': 5,
				'stellate': 6,
				'dcn': 7
				}

# Colors for plots (can be changed without constraints)
cell_color = {'golgi': '#332EBC',
			  'glomerulus': '#0E1030',
			  'granule': '#E62214',
			  'purkinje': '#0F8944',
			  'stellate': '#876506',
			  'basket': '#7A1607',
			  'dcn': '#15118B'}

# Define pc and dcn values once volume base size has been defined
pc_in_volume = int(volume_base_size[0]*volume_base_size[1]*cells_density['purkinje'])
dcn_in_volume = int(pc_in_volume / pc_dcn_ratio)
cells_density['dcn'] = dcn_in_volume / (dcn_volume[0]*dcn_volume[1]*layers_thick['dcn'])

dcn_volume = volume_base_size / 2
save_name = '{}_{}x{}_v3.hdf5'.format(filename, volume_base_size[0], volume_base_size[1])



### Must be generated from previous dictionaries!
# Store positions of cells - organized by cell type
final_cell_positions = {key: [] for key in cell_type_ID.keys()}
placement_stats = {key: {} for key in cell_type_ID.keys()}
for key, subdic in placement_stats.items():
	subdic['number_of_cells'] = []
	subdic['total_n_{}'.format(key)] = 0
	if key != 'purkinje':
		subdic['{}_subl'.format(key)] = 0

########################################################################################
############################### CONNECTOME PARAMETERS ##################################

# GoC parameters
r_goc_vol = 50	# radius of the GoC volume around the soma
# GoC axon
GoCaxon_z = 30		# max width of GoC axon (keep the minimum possible)
GoCaxon_y = 150		# max height (height of the total simulation volume)
GoCaxon_x = 150		# max lenght

# GrC and parallel fibers parameters
dend_len = 40		# maximum lenght of a GrC dendrite
h_m = 151		# offset for the height of each parallel fiber
sd = 66			# standard deviation of the parallel fibers distribution of heights

# basket and stellate cells parameters
r_sb = 15		# radius of stellate and basket cells area around soma


# Connectivity parameters for granular layer
n_conn_goc = 40		# =n_conn_gloms: number of glomeruli connected to a GoC
n_conn_glom = 4		# number of GoC connected to a glomerulus and number of glomeruli connected to the same GrC
n_connAA = 400		# number of ascending axons connected to a GoC == number of local parallel fibers connected to a GoC
n_conn_pf = 1200	# number of external parallel fibers connected to a GoC
tot_conn = 1600		# total number of conncetions between a GoC and parallel fibers

# Connectivity parameters for granular-molecular layers
# thresholds on inter-soma distances for bc/sc - PCs connectivity
distx = 500
distz = 100
#connectivity parameters
div = 2
conv = 20
# thresholds on inter-soma distances for gap junctions
d_xy = 150
d_z = 50
dc_gj = 4	# divergence = convergence in gap junctions connectivity
# maximum number of connections between parallel fibers and stellate, basket and Purkinje cells
max_conv_sc = 500	# convergence on stellate cells
max_conv_bc = 500	# convergence on basket cells
max_conv_pc = 10000	# convergence on Purkinje cells

# Connectivity parameters for PC-DCN layers
div_pc = 5	# maximum number of connections per PC (typically there are 4-5)
### Data for glom - dcn connectivity
conv_dcn = 147	# convergence
div_dcn = 2	# or 3 - to be tested - divergence
