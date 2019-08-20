import abc
from .helpers import ConfigurableClass
import numpy as np
from pprint import pprint

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
		oob = self.scaffold.configuration.X * 1000.
		n_connAA = self.aa_convergence
		n_conn_pf = self.pf_convergence
		tot_conn = n_connAA + n_conn_pf

		def connectome_grc_goc(first_granule, granules, golgicells, r_goc_vol, OoB_value, n_connAA, n_conn_pf, tot_conn):
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

		result_aa, result_pf = connectome_grc_goc(first_granule, granules, golgis, r_goc_vol, oob, n_connAA, n_conn_pf, tot_conn)
		self.scaffold.connect_cells(self, result_aa)
		self.scaffold.connect_cells(self, result_pf)
