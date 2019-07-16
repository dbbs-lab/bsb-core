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
volume_base_size = np.array([config.X * 10 ** 6, config.Z * 10 ** 6])
dcn_volume = volume_base_size / 2

# Name of data file
filename = 'scaffold_full_IO'
save_name = '{}_{}x{}_v3.hdf5'.format(filename, volume_base_size[0], volume_base_size[1])

# Purkinje / DCN ratio: the number of PC per DCN - (Note: describe better)
pc_dcn_ratio = 1. / config.CellTypes['DCN Cell'].ratio

# Extension of Purkinje cell dendritic tree
pc_extension_dend_tree = config.Geometries['PurkinjeCellGeometry'].tree_extension_x
z_pc = config.Geometries['PurkinjeCellGeometry'].tree_extension_z

# Thickness of different layers
layers_thick = {'granular': config.Layers['Granular Layer'].dimensions[1],
				'purkinje': config.Layers['Purkinje Layer'].dimensions[1],
				'molecular': config.Layers['Molecular Layer'].dimensions[1],
				'dcn': config.Layers['DCN Layer'].dimensions[1]}

# Soma radius of each cell type (micron)
''' Diameter of DCN Glutamatergic neuron is in range 15 - 35 micron (Aizemann et al., 2003)
	==> mean diam = 25 micron
	==> mean radius = 12.5 micron
	Slightly different estimate (Gauck and Jaeger, 2000): 10 - 35 micron, average = 20
	==> mean radius = 10 micron'''

cells_radius = {'golgi': config.CellTypes['Golgi Cell'].radius,
				'glomerulus': config.CellTypes['Glomerulus'].radius,
				'granule': config.CellTypes['Granule Cell'].radius,
				'purkinje': config.CellTypes['Purkinje Cell'].radius,
				'stellate': config.CellTypes['Stellate Cell'].radius,
				'basket': config.CellTypes['Basket Cell'].radius,
				'dcn': config.CellTypes['DCN Cell'].radius}


# Density distribution of each cell type
cells_density = {'golgi': config.CellTypes['Golgi Cell'].density,
			     'glomerulus': config.CellTypes['Glomerulus'].density,
			     'granule': config.CellTypes['Granule Cell'].density,
			     'purkinje': config.CellTypes['Purkinje Cell'].density,
			     'stellate': config.CellTypes['Stellate Cell'].density,
			     'basket': config.CellTypes['Basket Cell'].density}

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
cell_color = {  'golgi': config.CellTypes['Golgi Cell'].color,
				'glomerulus': config.CellTypes['Glomerulus'].color,
				'granule': config.CellTypes['Granule Cell'].color,
				'purkinje': config.CellTypes['Purkinje Cell'].color,
				'stellate': config.CellTypes['Stellate Cell'].color,
				'basket': config.CellTypes['Basket Cell'].color,
				'dcn': config.CellTypes['DCN Cell'].color}

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
r_goc_vol = config.Geometries['GolgiCellGeometry'].dendrite_radius	# radius of the GoC volume around the soma
# GoC axon
GoCaxon_z = config.Geometries['GolgiCellGeometry'].axon_z		# max width of GoC axon (keep the minimum possible)
GoCaxon_y = config.Geometries['GolgiCellGeometry'].axon_y		# max height (height of the total simulation volume)
GoCaxon_x = config.Geometries['GolgiCellGeometry'].axon_x		# max lenght

# GrC and parallel fibers parameters
dend_len = config.Geometries['GranuleCellGeometry'].dendrite_length		# maximum lenght of a GrC dendrite
h_m = config.Geometries['GranuleCellGeometry'].pf_height		# offset for the height of each parallel fiber
sd = config.Geometries['GranuleCellGeometry'].pf_height_sd			# standard deviation of the parallel fibers distribution of heights

# basket and stellate cells parameters
r_sb = config.Geometries['StellateCellGeometry'].radius		# radius of stellate and basket cells area around soma


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
