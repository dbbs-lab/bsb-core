import numpy as np

try:
	scaffoldInstance
except Exception as e:
	raise Exception("A scaffold instance needs to be present in the namespace when importing scaffold_params.py")

config = scaffoldInstance.configuration
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
volume_base_size = np.array([config.X, config.Z])
dcn_volume = volume_base_size / 2

# Name of data file
filename = 'scaffold_full_IO'
save_name = '{}_{}x{}_v3.hdf5'.format(filename, volume_base_size[0], volume_base_size[1])

# Purkinje / DCN ratio: the number of PC per DCN - (Note: describe better)
pc_dcn_ratio = 1. / config.cell_types['DCN Cell'].ratio

# Extension of Purkinje cell dendritic tree
pc_extension_dend_tree = float(config.Geometries['PurkinjeCellGeometry'].tree_extension_x)
z_pc = float(config.Geometries['PurkinjeCellGeometry'].tree_extension_z)

# Thickness of different layers
layers_thick = {'granular': config.layers['Granular Layer'].dimensions[1],
				'purkinje': config.layers['Purkinje Layer'].dimensions[1],
				'molecular': config.layers['Molecular Layer'].dimensions[1],
				'dcn': config.layers['DCN Layer'].dimensions[1]}

# Soma radius of each cell type (micron)
''' Diameter of DCN Glutamatergic neuron is in range 15 - 35 micron (Aizemann et al., 2003)
	==> mean diam = 25 micron
	==> mean radius = 12.5 micron
	Slightly different estimate (Gauck and Jaeger, 2000): 10 - 35 micron, average = 20
	==> mean radius = 10 micron'''

cells_radius = {'golgi': config.cell_types['Golgi Cell'].radius,
				'glomerulus': config.cell_types['Glomerulus'].radius,
				'granule': config.cell_types['Granule Cell'].radius,
				'purkinje': config.cell_types['Purkinje Cell'].radius,
				'stellate': config.cell_types['Stellate Cell'].radius,
				'basket': config.cell_types['Basket Cell'].radius,
				'dcn': config.cell_types['DCN Cell'].radius}


# Density distribution of each cell type
cells_density = {'golgi': config.cell_types['Golgi Cell'].density,
			     'glomerulus': config.cell_types['Glomerulus'].density,
			     'granule': config.cell_types['Granule Cell'].density,
			     'purkinje': config.cell_types['Purkinje Cell'].density,
			     'stellate': config.cell_types['Stellate Cell'].density,
			     'basket': config.cell_types['Basket Cell'].density}

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
cell_color = {  'golgi': config.cell_types['Golgi Cell'].color,
				'glomerulus': config.cell_types['Glomerulus'].color,
				'granule': config.cell_types['Granule Cell'].color,
				'purkinje': config.cell_types['Purkinje Cell'].color,
				'stellate': config.cell_types['Stellate Cell'].color,
				'basket': config.cell_types['Basket Cell'].color,
				'dcn': config.cell_types['DCN Cell'].color}

# Define pc and dcn values once volume base size has been defined
pc_in_volume = int(volume_base_size[0]*volume_base_size[1]*cells_density['purkinje'])
dcn_in_volume = int(pc_in_volume / pc_dcn_ratio)
cells_density['dcn'] = dcn_in_volume / (dcn_volume[0]*dcn_volume[1]*layers_thick['dcn'])

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
r_goc_vol = float(config.Geometries['GolgiCellGeometry'].dendrite_radius)	# radius of the GoC volume around the soma
# GoC axon
GoCaxon_z = float(config.Geometries['GolgiCellGeometry'].axon_z)		# max width of GoC axon (keep the minimum possible)
GoCaxon_y = float(config.Geometries['GolgiCellGeometry'].axon_y)		# max height (height of the total simulation volume)
GoCaxon_x = float(config.Geometries['GolgiCellGeometry'].axon_x)		# max lenght

# GrC and parallel fibers parameters
dend_len = float(config.Geometries['GranuleCellGeometry'].dendrite_length)		# maximum lenght of a GrC dendrite
h_m = float(config.Geometries['GranuleCellGeometry'].pf_height)		# offset for the height of each parallel fiber
sd = float(config.Geometries['GranuleCellGeometry'].pf_height_sd)			# standard deviation of the parallel fibers distribution of heights

# basket and stellate cells parameters
r_sb = float(config.Geometries['StellateCellGeometry'].radius)		# radius of stellate and basket cells area around soma


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
