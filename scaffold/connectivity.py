import abc
from .helpers import ConfigurableClass
from .postprocessing import get_parallel_fiber_heights, get_dcn_rotations
import numpy as np

class ConnectionStrategy(ConfigurableClass):
	@abc.abstractmethod
	def connect(self):
		pass

class ReciprocalGolgiGlomerulus(ConnectionStrategy):
	def validate(self):
		pass

	def connect(self):
		pass

class TouchingConvergenceDivergence(ConnectionStrategy):
	casts = {
		'divergence': int,
		'convergence': int
	}

	required = ['divergence', 'convergence']

	def validate(self):
		pass

	def connect(self):
		pass

class TouchConnect(ConnectionStrategy):
	def validate(self):
		pass

	def connect(self):
		pass

class ConnectomeGlomerulusGranule(TouchingConvergenceDivergence):
	'''
		Legacy implementation for the connections between glomeruli and granule cells.
	'''
	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		from_celltype = self.from_celltype
		to_celltype = self.to_celltype
		glomeruli = self.scaffold.cells_by_type[from_celltype.name]
		granules = self.scaffold.cells_by_type[to_celltype.name]
		dend_len = to_celltype.geometry.dendrite_length
		n_conn_glom = self.convergence
		first_glomerulus = int(glomeruli[0,0])

		def connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom):
			'''
				Legacy code block to connect glomeruli to granule cells
			'''
			glom_x = glomeruli[:,2]
			glom_y = glomeruli[:,3]
			glom_z = glomeruli[:,4]
			results = np.empty((granules.shape[0] * n_conn_glom, 2))
			next_index = 0
			# Find glomeruli to connect to each granule cell
			for gran_id, gran_type, gran_x, gran_y, gran_z in granules:
				# Use a naive approach to find all glomeruli at a maximum distance of `dendrite_length`
				distance_vector = ((glom_x-gran_x)**2)+((glom_y-gran_y)**2)+((glom_z-gran_z)**2)-(dend_len**2)
				good_gloms = np.where((distance_vector < 0.) == True)[0]	# indexes of glomeruli that can potentially be connected
				good_gloms_len = len(good_gloms)
				# Do we find more than enough candidates?
				if good_gloms_len > n_conn_glom: # Yes: select the closest ones
					# Get the distances of the glomeruli within range
					gloms_distance = distance_vector[good_gloms]
					# Sort the good glomerulus id vector by the good glomerulus distance vector
					connected_gloms = good_gloms[gloms_distance.argsort()]
					connected_glom_len = n_conn_glom
				else: # No: select all of them
					connected_gloms = good_gloms
					connected_glom_len = good_gloms_len
				# Connect the selected glomeruli to the current gran_id
				for i in range(connected_glom_len):
					# Add the first_glomerulus id to convert their local id to their real simulation id
					results[next_index + i] = [connected_gloms[i] + first_glomerulus, gran_id]
				# Move up the internal array pointer
				next_index += connected_glom_len
			# Truncate the pre-allocated array to the internal array pointer.
			return results[0:next_index,:]

		# Execute legacy code and add the connection matrix it returns to the scaffold.
		connectome = connectome_glom_grc(first_glomerulus, glomeruli, granules, dend_len, n_conn_glom)
		self.scaffold.connect_cells(self, connectome)

class ConnectomeGlomerulusGolgi(TouchingConvergenceDivergence):
	'''
		Legacy implementation for the connections between Golgi cells and glomeruli.
	'''
	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		glomerulus_celltype = self.from_celltype
		golgi_celltype = self.to_celltype
		glomeruli = self.scaffold.cells_by_type[glomerulus_celltype.name]
		golgis = self.scaffold.cells_by_type[golgi_celltype.name]
		first_glomerulus = int(glomeruli[0,0])
		r_goc_vol = golgi_celltype.geometry.dendrite_radius

		def connectome_glom_goc(first_glomerulus, glomeruli, golgicells, r_goc_vol):
			glom_bd = np.zeros((0,2))
			glom_x = glomeruli[:,2]
			glom_y = glomeruli[:,3]
			glom_z = glomeruli[:,4]

			# for all Golgi cells: calculate which glomeruli fall into the volume of GoC basolateral dendrites, then choose 40 of them for the connection and delete them from successive computations, since 1 axon is connected to 1 GoC
			for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in golgicells:

				# Geometric constraints: glom less than `r_goc_vol` away from golgi and golgi cell soma above glom.
				volume_matrix = (((glom_x - golgi_x)**2)+((glom_y-golgi_y)**2)+((glom_z-golgi_z)**2)-(r_goc_vol**2)).__le__(0) & (glom_y).__le__(golgi_y)
				good_gloms = np.where(volume_matrix==True)[0]	# finds indexes of granules that can potentially be connected
				connected_gloms = good_gloms + first_glomerulus # Translate local id to simulation id

				matrix = np.zeros((len(good_gloms), 2))
				matrix[:,0] = connected_gloms # from cell
				matrix[:,1] = golgi_id	# to cell
				glom_bd = np.vstack((glom_bd, matrix))

			return glom_bd

		connectome = connectome_glom_goc(first_glomerulus, glomeruli, golgis, r_goc_vol)
		self.scaffold.connect_cells(self, connectome)

class ConnectomeGolgiGlomerulus(TouchingConvergenceDivergence):
	'''
		Legacy implementation for the connections between glomeruli and Golgi cells.
	'''
	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		golgi_celltype = self.from_celltype
		glomerulus_celltype = self.to_celltype
		glomeruli = self.scaffold.cells_by_type[glomerulus_celltype.name]
		golgis = self.scaffold.cells_by_type[golgi_celltype.name]
		first_glomerulus = int(glomeruli[0,0])
		GoCaxon_x = golgi_celltype.geometry.axon_x
		GoCaxon_y = golgi_celltype.geometry.axon_y
		GoCaxon_z = golgi_celltype.geometry.axon_z
		r_glom = glomerulus_celltype.radius
		n_conn_goc = self.divergence
		layer_thickness = self.scaffold.configuration.get_layer(name=golgi_celltype.placement.layer).thickness
		# An arbitrarily large value that will be used to exclude cells from geometric constraints
		oob = self.scaffold.configuration.X * 1000.
		def connectome_goc_glom(first_glomerulus, glomeruli, golgicells, GoCaxon_x, GoCaxon_y, GoCaxon_z, r_glom, n_conn_goc, layer_thickness, oob):
			glom_x = glomeruli[:,2]
			glom_y = glomeruli[:,3]
			glom_z = glomeruli[:,4]
			new_glomeruli = np.copy(glomeruli)
			new_golgicells = np.random.permutation(golgicells)
			connections = np.zeros((golgis.shape[0] * n_conn_goc ,2))
			new_connection_index = 0

			# for all Golgi cells: calculate which glomeruli fall into the area of GoC axon, then choose 40 of them for the connection and delete them from successive computations, since 1 glomerulus must be connected to only 1 GoC
			for golgi_id, golgi_type, golgi_x, golgi_y, golgi_z in new_golgicells:
				# Check geometrical constraints
				# glomerulus falls into the x range of values?
				bool_vector = (((glom_x+r_glom).__ge__(golgi_x-GoCaxon_x/2.)) & ((glom_x-r_glom).__le__(golgi_x+GoCaxon_x/2.)))
				# glomerulus falls into the y range of values?
				bool_vector = bool_vector & (((glom_y+r_glom).__ge__(golgi_y-GoCaxon_y/2.)) & ((glom_y-r_glom).__le__(golgi_y+GoCaxon_y/2.)))
				# glomerulus falls into the z range of values?
				bool_vector = bool_vector & (((glom_z+r_glom).__ge__(golgi_z-GoCaxon_z/2.)) & ((glom_z-r_glom).__le__(golgi_z+GoCaxon_z/2.)))

				# Make a permutation of all candidate glomeruli
				good_gloms = np.where(bool_vector)[0]
				chosen_rand = np.random.permutation(good_gloms)
				good_gloms_matrix = new_glomeruli[chosen_rand]
				# Calculate the distance between the golgi cell and all glomerulus candidates, normalize distance by layer thickness
				normalized_distance_vector = np.sqrt((good_gloms_matrix[:,2]-golgi_x)**2 + (good_gloms_matrix[:,3]-golgi_y)**2) / layer_thickness
				sorting_map = normalized_distance_vector.argsort()
				# Sort the candidate glomerulus matrix and distance vector by the distance vector
				good_gloms_matrix = good_gloms_matrix[sorting_map]
				# Use the normalized distance vector as a probability treshold for connecting glomeruli
				probability_treshold = normalized_distance_vector[sorting_map]

				idx = 1
				for candidate_index, glomerulus in enumerate(good_gloms_matrix):
					if idx <= n_conn_goc:
						ra = np.random.random()
						if (ra).__gt__(probability_treshold[candidate_index]):
							glomerulus_id = glomerulus[0]
							connections[new_connection_index, 0] = golgi_id
							connections[new_connection_index, 1] = glomerulus_id + first_glomerulus
							new_glomeruli[int(glomerulus_id - first_glomerulus),:] = oob
							new_connection_index += 1
							idx += 1
			return connections[0:new_connection_index]

		result = connectome_goc_glom(first_glomerulus, glomeruli, golgis, GoCaxon_x, GoCaxon_y, GoCaxon_z, r_glom, n_conn_goc, layer_thickness, oob)
		self.scaffold.connect_cells(self, result)

class ConnectomeGranuleGolgi(ConnectionStrategy):
	'''
		Legacy implementation for the connections between Golgi cells and glomeruli.
	'''

	casts = {
		'aa_convergence': int,
		'pf_convergence': int
	}

	required = ['aa_convergence', 'pf_convergence']

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		granule_celltype = self.from_celltype
		golgi_celltype = self.to_celltype
		granules = self.scaffold.cells_by_type[granule_celltype.name]
		golgis = self.scaffold.cells_by_type[golgi_celltype.name]
		first_granule = int(granules[0, 0])
		r_goc_vol = golgi_celltype.geometry.dendrite_radius
		oob = self.scaffold.configuration.X * 1000. # Any arbitrarily large value outside of simulation volume
		n_connAA = self.aa_convergence
		n_conn_pf = self.pf_convergence
		tot_conn = n_connAA + n_conn_pf
		pf_heights = get_parallel_fiber_heights(self.scaffold, granule_celltype.geometry, granules)
		self.scaffold.append_dset('hpf', data=pf_heights)

		def connectome_grc_goc(first_granule, granules, golgicells, r_goc_vol, OoB_value, n_connAA, n_conn_pf, tot_conn, scaffold):
			aa_goc = np.empty((0,2))
			pf_goc = np.empty((0,2))
			densityWarningSent = False
			new_granules = np.copy(granules)
			granules_x = new_granules[:,2]
			granules_z = new_granules[:,4]
			new_golgicells = np.random.permutation(golgicells)
			if new_granules.shape[0] <= new_golgicells.shape[0]:
				raise Exception("The number of granule cells was less than the number of golgi cells. Simulation cannot continue.")
			for golgi_id, _, golgi_x, golgi_y, golgi_z in new_golgicells:
				# Distance of this golgi cell to all ascending axons
				distance_vector = ((granules_x-golgi_x)**2)+((granules_z-golgi_z)**2)
				AA_candidates = np.where((distance_vector).__le__(r_goc_vol**2))[0]		# finds indexes of ascending axons that can potentially be connected
				chosen_rand = np.random.permutation(AA_candidates)
				selected_granules = new_granules[chosen_rand]
				selected_distances = np.sqrt(distance_vector[chosen_rand])
				prob = selected_distances / r_goc_vol
				distance_sort = prob.argsort()
				selected_granules = selected_granules[distance_sort]
				prob = prob[distance_sort]
				rolls = np.random.uniform(size=len(selected_granules))
				connectedAA = np.empty(n_connAA)
				idx = 0
				for ind,j in enumerate(selected_granules):
					if idx < n_connAA:
						if rolls[ind] > prob[ind]:
							connectedAA[idx] = j[0]
							idx += 1
				connectedAA = connectedAA[0:idx]
				good_grc = np.delete(granules, (connectedAA - first_granule), 0)
				intersections = (good_grc[:,2]).__ge__(golgi_x-r_goc_vol) & (good_grc[:,2]).__le__(golgi_x+r_goc_vol)
				good_pf = np.where(intersections==True)[0]				# finds indexes of granules that can potentially be connected
				# The remaining amount of parallel fibres to connect after subtracting the amount of already connected ascending axons.
				AA_connected_count = len(connectedAA)
				parallelFibersToConnect = tot_conn - AA_connected_count
				# Randomly select parallel fibers to be connected with a GoC, to a maximum of tot_conn connections
				if good_pf.shape[0] < parallelFibersToConnect:
					connected_pf = np.random.choice(good_pf, min(tot_conn-AA_connected_count, good_pf.shape[0]), replace = False)
					totalConnectionsMade = connected_pf.shape[0] + AA_connected_count
					# Warn the user once if not enough granule cells are present to connect to the Golgi cell.
					if not densityWarningSent:
						densityWarningSent = True
						if scaffold.configuration.verbosity > 0:
							print("[WARNING] The granule cell density is too low compared to the Golgi cell density to make physiological connections!")
				else:
					connected_pf = np.random.choice(good_pf, tot_conn-len(connectedAA), replace = False)
					totalConnectionsMade = tot_conn
				PF_connected_count = connected_pf.shape[0]
				pf_idx = good_grc[connected_pf,:]
				matrix_aa = np.zeros((AA_connected_count, 2))
				matrix_pf = np.zeros((PF_connected_count, 2))
				matrix_pf[0:PF_connected_count, 0] = pf_idx[:,0]
				matrix_aa[0:AA_connected_count, 0] = connectedAA
				matrix_pf[:,1] = golgi_id
				matrix_aa[:,1] = golgi_id
				pf_goc = np.vstack((pf_goc, matrix_pf))
				aa_goc = np.vstack((aa_goc, matrix_aa))
				new_granules[((connectedAA.astype(int)) - first_granule),:] = OoB_value
				# End of Golgi cell loop
			aa_goc = aa_goc[aa_goc[:,1].argsort()]
			pf_goc = pf_goc[pf_goc[:,1].argsort()]		# sorting of the resulting vector on the post-synaptic neurons
			return aa_goc, pf_goc

		result_aa, result_pf = connectome_grc_goc(first_granule, granules, golgis, r_goc_vol, oob, n_connAA, n_conn_pf, tot_conn, self.scaffold)
		self.scaffold.connect_cells(self, result_aa, "AscendingAxonGolgi")
		self.scaffold.connect_cells(self, result_pf)

class ConnectomeGolgiGranule(ConnectionStrategy):
	'''
		Legacy implementation for the connections between Golgi cells and granule cells.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		glom_grc = self.scaffold.cell_connections_by_type['GlomerulusGranule']
		goc_glom = self.scaffold.cell_connections_by_type['GolgiGlomerulus']
		golgi_type = self.from_celltype
		golgis = self.scaffold.cells_by_type[golgi_type.name]

		def connectome_goc_grc(golgis, glom_grc, goc_glom):
			# Connect all golgi cells to the granule cells that they share a glomerulus with.
			glom_grc_per_glom = {}
			goc_grc = np.empty((0, 2))
			golgi_ids = golgis[:, 0]
			for golgi_id in golgis[:, 0]:
				# Fetch all the glomeruli this golgi is connected to
				connected_glomeruli = goc_glom[goc_glom[:, 0] == golgi_id, 1]
				# Append a new set of connections after the existing set of goc_grc connections.
				connected_granules_via_gloms = list(map(lambda row: row[1], filter(lambda row: row[0] in connected_glomeruli, glom_grc)))
				goc_grc = np.vstack((
					goc_grc,
					# Create a matrix with 2 columns where the 1st column is the golgi id
					np.column_stack((
						golgi_id * np.ones(len(connected_granules_via_gloms)),
						# and the 2nd column is all granules connected to one of the glomeruli the golgi is connected to.
						connected_granules_via_gloms
					))
				))

			return goc_grc


		result = connectome_goc_grc(golgis, glom_grc, goc_glom)
		self.scaffold.connect_cells(self, result)

class ConnectomeAscAxonPurkinje(ConnectionStrategy):
	'''
		Legacy implementation for the connections between ascending axons and purkinje cells.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		granule_celltype = self.from_celltype
		purkinje_celltype = self.to_celltype
		granules = self.scaffold.cells_by_type[granule_celltype.name]
		purkinjes = self.scaffold.cells_by_type[purkinje_celltype.name]
		first_granule = int(granules[0, 0])
		OoB_value = self.scaffold.configuration.X * 1000. # Any arbitrarily large value outside of simulation volume
		purkinje_extension_x = purkinje_celltype.placement.extension_x
		purkinje_extension_z = purkinje_celltype.placement.extension_z

		def connectome_aa_pc(first_granule, granules, purkinjes, x_pc, z_pc, OoB_value):
			aa_pc = np.zeros((0, 2))
			new_granules = np.copy(granules)

			# for all Purkinje cells: calculate and choose which granules fall into the area of PC dendritic tree, then delete them from successive computations, since 1 ascending axon is connected to only 1 PC
			for i in purkinjes:

				# ascending axon falls into the z range of values?
				bool_vector = (new_granules[:,4]).__ge__(i[4]-z_pc/2.) & (new_granules[:,4]).__le__(i[4]+z_pc/2.)
				# ascending axon falls into the x range of values?
				bool_vector = bool_vector & ((new_granules[:,2]).__ge__(i[2]-x_pc/2.) & (new_granules[:,2]).__le__(i[2]+x_pc/2.))
				good_aa = np.where(bool_vector)[0]	# finds indexes of ascending axons that, on the selected axis, have the correct sum value

				# construction of the output matrix: the first column has the GrC id, while the second column has the PC id
				matrix = np.zeros((len(good_aa), 2))
				matrix[:,1] = i[0]
				matrix[:,0] = good_aa + first_granule
				aa_pc = np.vstack((aa_pc, matrix))

				new_granules[good_aa,:] = OoB_value	# update the granules matrix used for computation by deleting the coordinates of connected ones


			return aa_pc

		result = connectome_aa_pc(first_granule, granules, purkinjes, purkinje_extension_x, purkinje_extension_z, OoB_value)
		self.scaffold.connect_cells(self, result, "AscAxonPurkinje")

class ConnectomePFPurkinje(ConnectionStrategy):
	'''
		Legacy implementation for the connections between parallel fibers and purkinje cells.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		granule_celltype = self.from_celltype
		purkinje_celltype = self.to_celltype
		granules = self.scaffold.cells_by_type[granule_celltype.name]
		purkinjes = self.scaffold.cells_by_type[purkinje_celltype.name]
		first_granule = int(granules[0, 0])
		purkinje_extension_x = purkinje_celltype.placement.extension_x

		def connectome_pf_pc(first_granule, granules, purkinjes, x_pc):
			pf_pc = np.zeros((0,2))
			# for all Purkinje cells: calculate and choose which parallel fibers fall into the area of PC dendritic tree (then delete them from successive computations, since 1 parallel fiber is connected to a maximum of PCs)
			for i in purkinjes:

				# which parallel fibers fall into the x range of values?
				bool_matrix = (granules[:,2]).__ge__(i[2]-x_pc/2.) & (granules[:,2]).__le__(i[2]+x_pc/2.)	# CAMBIARE IN new_granules SE VINCOLO SU 30 pfs
				good_pf = np.where(bool_matrix)[0]	# finds indexes of parallel fibers that, on the selected axis, satisfy the condition

				# construction of the output matrix: the first column has the GrC id, while the second column has the PC id
				matrix = np.zeros((len(good_pf), 2))
				matrix[:,1] = i[0]
				matrix[:,0] = good_pf + first_granule
				pf_pc = np.vstack((pf_pc, matrix))

			return pf_pc

		result = connectome_pf_pc(first_granule, granules, purkinjes, purkinje_extension_x)
		self.scaffold.connect_cells(self, result, "PFPurkinje")

class ConnectomePFInterneuron(ConnectionStrategy):
	'''
		Legacy implementation for the connections between parallel fibers and a molecular layer interneuron celltype.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		granule_celltype = self.from_celltype
		interneuron_celltype = self.to_celltype
		granules = self.scaffold.cells_by_type[granule_celltype.name]
		interneurons = self.scaffold.cells_by_type[interneuron_celltype.name]
		first_granule = int(granules[0, 0])
		dendrite_radius = interneuron_celltype.geometry.radius
		pf_heights = self.scaffold.appends['hpf'][:, 1] + granules[:, 3] # Add granule Y to height of its pf

		def connectome_pf_inter(first_granule, interneurons, granules, r_sb, h_pf):
			pf_interneuron = np.zeros((0,2))

			for i in interneurons:	# for each interneuron find all the parallel fibers that fall into the sphere with centre the cell soma and appropriate radius

				# find all cells that satisfy the condition
				interneuron_matrix = (((granules[:,2]-i[2])**2)+((h_pf-i[3])**2)-(r_sb**2)).__le__(0)
				good_pf = np.where(interneuron_matrix)[0]	# indexes of interneurons that can potentially be connected

				matrix = np.zeros((len(good_pf), 2))
				matrix[:,1] = i[0]
				matrix[:,0] = good_pf + first_granule
				pf_interneuron = np.vstack((pf_interneuron, matrix))

			return pf_interneuron

		result = connectome_pf_inter(first_granule, interneurons, granules, dendrite_radius, pf_heights)
		self.scaffold.connect_cells(self, result)

class ConnectomeBCSCPurkinje(ConnectionStrategy):
	'''
		Legacy implementation for the connections between stellate cells, basket cells and purkinje cells.
	'''

	casts = {
		'limit_x': float,
		'limit_z': float,
		'divergence': int,
		'convergence': int
	}

	required = ['limit_x', 'limit_z', 'divergence', 'convergence']

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		stellate_celltype = self.from_celltype
		basket_celltype = self.scaffold.configuration.cell_types["Basket Cell"]
		purkinje_celltype = self.to_celltype
		stellates = self.scaffold.cells_by_type[stellate_celltype.name]
		purkinjes = self.scaffold.cells_by_type[purkinje_celltype.name]
		baskets = self.scaffold.cells_by_type[basket_celltype.name]
		first_stellate = int(stellates[0, 0])
		first_basket = int(baskets[0, 0])
		distx = self.limit_x
		distz = self.limit_z
		conv = self.convergence

		def connectome_sc_bc_pc(first_stellate, first_basket, basketcells, stellates, purkinjes, distx, distz, conv):
			n_purkinje = len(purkinjes)
			sc_pc = np.empty((n_purkinje * conv, 2))
			bc_pc = np.empty((n_purkinje * conv, 2))
			bc_i = 0
			sc_i = 0

			stellates_x = stellates[:, 2]
			stellates_z = stellates[:, 4]
			baskets_x = baskets[:, 2]
			baskets_z = baskets[:, 4]
			for p_id, p_type, p_x, p_y, p_z in purkinjes:	# for all Purkinje cells: calculate which basket and stellate cells can be connected, then choose 20 of them for each typology

				idx_bc = 1
				idx_sc = 1

				# find all cells that satisfy the distance condition for both types
				sc_matrix = (np.absolute(stellates_z-p_z)).__lt__(distz) & (np.absolute(stellates_x-p_x)).__lt__(distx)
				bc_matrix = (np.absolute(baskets_z-p_z)).__lt__(distx) & (np.absolute(baskets_x-p_x)).__lt__(distz)


				good_bc = np.where(bc_matrix)[0]	# indexes of basket cells that can potentially be connected
				good_sc = np.where(sc_matrix)[0]	# indexes of stellate cells that can potentially be connected

				chosen_rand_bc = np.random.permutation(good_bc)
				good_bc_matrix = basketcells[chosen_rand_bc]
				chosen_rand_sc = np.random.permutation(good_sc)
				good_sc_matrix = stellates[chosen_rand_sc]


				# basket cells connectivity
				for j in good_bc_matrix:

					if idx_bc <= conv:

						ra = np.random.random()
						if (ra).__gt__((np.absolute(j[4]-p_z))/distx) & (ra).__gt__((np.absolute(j[2]-p_x))/distz):

							idx_bc += 1
							bc_pc[bc_i, 0] = j[0]
							bc_pc[bc_i, 1] = p_id
							bc_i += 1


				# stellate cells connectivity
				for k in good_sc_matrix:

					if idx_sc <= conv:

						ra = np.random.random()
						if (ra).__gt__((np.absolute(k[4]-p_z))/distz) & (ra).__gt__((np.absolute(k[2]-p_x))/distx):

							idx_sc += 1
							sc_pc[sc_i, 0] = k[0]
							sc_pc[sc_i, 1] = p_id
							sc_i += 1


			return sc_pc[0:sc_i], bc_pc[0:bc_i]

		result_sc, result_bc = connectome_sc_bc_pc(first_stellate, first_basket, baskets, stellates, purkinjes, distx, distz, conv)
		self.scaffold.connect_cells(self, result_sc, "StellatePurkinje")
		self.scaffold.connect_cells(self, result_bc, "BasketPurkinje")

class ConnectomeGapJunctions(ConnectionStrategy):
	'''
		Legacy implementation for gap junctions between a cell type.
	'''

	casts = {
		'limit_xy': float,
		'limit_z': float,
		'divergence': int
	}

	required = ['limit_xy', 'limit_z', 'divergence']

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		from_celltype = self.from_celltype
		from_cells = self.scaffold.cells_by_type[from_celltype.name]
		first_cell = int(from_cells[0, 0])
		limit_xy = self.limit_xy
		limit_z = self.limit_z
		divergence = self.divergence

		def gap_junctions(cells, d_xy, d_z, dc_gj):
			gj_sc = np.zeros((cells.shape[0] * dc_gj ,2))
			gj_i = 0
			cells_x = cells[:,2]
			cells_y = cells[:,3]
			cells_z = cells[:,4]

			for id, type, x, y, z in cells:	# for each stellate cell calculate the distance with every other cell of the same type in the volume, then choose 4 of them

				idx = 1

				# find all cells that satisfy the distance condition
				constraint_vector = (np.absolute(cells_z-z)).__lt__(d_z) & (np.absolute(cells_z-z)).__ne__(0) & (np.sqrt((cells_x-x)**2 + (cells_y-y)**2)).__lt__(d_xy)
				good_sc = np.where(constraint_vector)[0]	# indexes of stellate cells that can potentially be connected
				chosen_rand = np.random.permutation(good_sc)
				candidates = cells[chosen_rand]

				for j in candidates:

					if idx <= dc_gj:

						ra = np.random.random()
						if (ra).__gt__((np.absolute(j[4]-z))/float(d_z)) & (ra).__gt__((np.sqrt((j[2]-x)**2 + (j[3]-y)**2))/float(d_xy)):

							idx += 1
							gj_sc[gj_i, 0] = id
							gj_sc[gj_i, 1] = j[0]
							gj_i += 1

			return gj_sc[0:gj_i]

		result = gap_junctions(from_cells, limit_xy, limit_z, divergence)
		self.scaffold.connect_cells(self, result)

class ConnectomeGapJunctionsGolgi(ConnectionStrategy):
	'''
		Legacy implementation for Golgi cell gap junctions.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		golgi_celltype = self.from_celltype
		golgis = self.scaffold.cells_by_type[golgi_celltype.name]
		first_golgi = int(golgis[0, 0])
		r_goc_vol = golgi_celltype.geometry.dendrite_radius
		GoCaxon_x = golgi_celltype.geometry.axon_x
		GoCaxon_y = golgi_celltype.geometry.axon_y
		GoCaxon_z = golgi_celltype.geometry.axon_z

		def connectome_gj_goc(r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgicells, first_golgi):
			gj_goc = np.zeros((golgis.shape[0] ** 2,2))
			start_index = 0
			golgi_x = golgicells[:, 2]
			golgi_y = golgicells[:, 3]
			golgi_z = golgicells[:, 4]
			self_index = 0
			for i in golgicells:	# for each Golgi find all cells of the same type that, through their dendritic tree, fall into its axonal tree

				bool_vector = (np.absolute(golgi_x-i[2])).__le__(r_goc_vol + (GoCaxon_x/2.))
				bool_vector[self_index] = False # Don't connect a golgi cell to itself
				bool_vector = bool_vector & (np.absolute(golgi_y-i[3])).__le__(r_goc_vol + (GoCaxon_y/2.))
				bool_vector = bool_vector & (np.absolute(golgi_z-i[4])).__le__(r_goc_vol + (GoCaxon_z/2.))

				good_goc = np.where(bool_vector)[0]	# finds indexes of Golgi cells that satisfy all conditions

				end_index = start_index + len(good_goc)
				gj_goc[start_index:end_index, 0] = i[0]
				gj_goc[start_index:end_index, 1] = good_goc
				start_index = end_index
				self_index += 1

			return gj_goc[0:end_index]

		result = connectome_gj_goc(r_goc_vol, GoCaxon_x, GoCaxon_y, GoCaxon_z, golgis, first_golgi)
		self.scaffold.connect_cells(self, result)


class ConnectomePurkinjeDCN(ConnectionStrategy):
	'''
		Legacy implementation for the connection between purkinje cells and DCN cells.
		Also rotates the dendritic trees of the DCN.
	'''

	casts = {
		'divergence': int
	}

	required = ['divergence']

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		purkinje_celltype = self.from_celltype
		dcn_celltype = self.to_celltype
		purkinjes = self.scaffold.cells_by_type[purkinje_celltype.name]
		dcn_cells = self.scaffold.cells_by_type[dcn_celltype.name]
		dcn_angles = get_dcn_rotations(dcn_cells)
		self.scaffold.append_dset("DCNangle", data=dcn_angles)
		if len(dcn_cells) == 0:
			return
		first_dcn = int(dcn_cells[0,0])
		divergence = self.divergence

		def connectome_pc_dcn(first_dcn, purkinjes, dcn_cells, div_pc, dend_tree_coeff):
			pc_dcn = np.zeros((0, 2))

			for i in purkinjes:	# for all Purkinje cells: calculate the distance with the area around glutamatergic DCN cells soma, then choose 4-5 of them

				distance = np.zeros((dcn_cells.shape[0]))
				distance = (np.absolute((dend_tree_coeff[:,0]*i[2])+(dend_tree_coeff[:,1]*i[3])+(dend_tree_coeff[:,2]*i[4])+dend_tree_coeff[:,3]))/(np.sqrt((dend_tree_coeff[:,0]**2)+(dend_tree_coeff[:,1]**2)+(dend_tree_coeff[:,2]**2)))

				dist_matrix = np.zeros((dcn_cells.shape[0],2))
				dist_matrix[:,1] = dcn_cells[:,0]
				dist_matrix[:,0] = distance
				dcn_dist = np.random.permutation(dist_matrix)

				# If the number of DCN cells are less than the divergence value, all neurons are connected to the corresponding PC
				if dcn_cells.shape[0]<div_pc:
					matrix = np.zeros((dcn_cells.shape[0], 2))
					matrix[:,0] = i[0]
					matrix[:,1] = dcn_cells[:, 0]
					pc_dcn = np.vstack((pc_dcn, matrix))

				else:
					if np.random.rand()>0.5:

						connected_f = dcn_dist[0:div_pc,1]
						connected_dist = dcn_dist[0:div_pc,0]
						connected_provv = connected_f.astype(int)
						connected_dcn = connected_provv

						# construction of the output matrix: the first column has the  PC index, while the second column has the connected DCN cell index
						matrix = np.zeros((div_pc, 2))
						matrix[:,0] = i[0]
						matrix[:,1] = connected_dcn
						pc_dcn = np.vstack((pc_dcn, matrix))

					else:
						connected_f = dcn_dist[0:(div_pc-1),1]
						connected_dist = dcn_dist[0:(div_pc-1),0]
						connected_provv = connected_f.astype(int)
						connected_dcn = connected_provv

						matrix = np.zeros(((div_pc-1), 2))
						matrix[:,0] = i[0]
						matrix[:,1] = connected_dcn
						pc_dcn = np.vstack((pc_dcn, matrix))
			return pc_dcn

		results = connectome_pc_dcn(first_dcn, purkinjes, dcn_cells, divergence, dcn_angles)
		self.scaffold.connect_cells(self, results)


class ConnectomeGlomDCN(TouchingConvergenceDivergence):
	'''
		Legacy implementation for the connection between glomeruli and DCN cells.
	'''

	def validate(self):
		pass

	def connect(self):
		# Gather information for the legacy code block below.
		glomerulus_celltype = self.from_celltype
		dcn_celltype = self.to_celltype
		glomeruli = self.scaffold.cells_by_type[glomerulus_celltype.name]
		dcn_cells = self.scaffold.cells_by_type[dcn_celltype.name]
		first_glomerulus = int(glomeruli[0, 0])
		convergence = self.convergence

		def connectome_glom_dcn(first_glomerulus, glomeruli, dcn_glut, conv_dcn):
			glom_dcn = np.zeros((0, 2))
			for i in dcn_glut:

				connected_gloms = np.random.choice(glomeruli[:,0], conv_dcn, replace=False)

				matrix = np.zeros((conv_dcn, 2))
				matrix[:,0] = connected_gloms.astype(int)
				matrix[:,1] = i[0]

				glom_dcn = np.vstack((glom_dcn, matrix))

			return glom_dcn

		results = connectome_glom_dcn(first_glomerulus, glomeruli, dcn_cells, convergence)
		self.scaffold.connect_cells(self, results)
