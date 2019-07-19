import abc
from .helpers import ConfigurableClass
from pprint import pprint
import numpy as np

class PlacementStrategy(ConfigurableClass):
	@abc.abstractmethod
	def place(self, scaffold, cellType):
		pass

class LayeredRandomWalk(PlacementStrategy):
	'''
		Implementation of the placement of cells in sublayers via a self avoiding random walk.
	'''

	casts = {
		'distance_multiplier_min': float,
		'distance_multiplier_max': float
	}

	defaults = {
		'distance_multiplier_min': 3.,
		'distance_multiplier_max': 5.
	}

	def initialise(self, scaffold):
		super().initialise(scaffold)
		self.layerObject = scaffold.configuration.Layers[self.layer]


	def validate(self):
		# Check if the layer is given and exists.
		config = self.scaffold.configuration
		if not hasattr(self, 'layer'):
			raise Exception("Required attribute Layer missing from {}".format(self.name))
		if not self.layer in config.Layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))

	def place(self, cellType):
		'''
			Cell placement.

			1. Sublayer partitioning:
				Divide the layer into Y-stacked sublayers dependent on the volume still available in the
				layer, the cell radius and the cell density.
			2. Cell placement:

		'''
		config = self.scaffold.configuration
		layer = self.layerObject
		layerThickness = layer.thickness
		# Calculate the number of cells that belong in the available volume, rounded down.
		cellType.nToPlace = int(layer.availableVolume * cellType.density)
		if cellType.nToPlace == 0:
			print("[WARNING] Volume or density too low, no '{}' cells will be placed".format(cellType.name))
			nSublayers = 1
			cellType.ϵ = 0.
		else:
			# Calculate the volume available per cell
			cellType.placementVolume = layer.availableVolume / cellType.nToPlace
			# Calculate the radius of that volume's sphere
			cellType.placementRadius = (0.75 * cellType.placementVolume / np.pi) ** (1. / 3.0)
			# Calculate the cell epsilon: This is the length of the 'spare space' a cell has inside of its volume
			cellType.ϵ = cellType.placementRadius - cellType.radius
			# Calculate the amount of sublayers
			nSublayers = np.round(layerThickness / (1.5 * cellType.placementRadius))

		## Sublayer partitioning
		partitions = self.partitionLayer(nSublayers)
		# Adjust partitions for cell radius.
		partitions = partitions + np.array([cellType.radius, -cellType.radius])

		## Placement
		min_mult = self.distance_multiplier_min
		max_mult = self.distance_multiplier_max
		cellsPerSublayer = np.round(cellType.nToPlace / nSublayers)
		# if cell_type == 'granule':
		# 	min_mult, max_mult = 2., 2.
		# else:
		# 	min_mult, max_mult = 3., 5.

		for subl in np.arange(nSublayers):
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
			print( "{} sublayer number {} out of {} filled".format(cell_type, subl+1, nSublayers))

		matrix_reframe = np.empty((1,3))
		for subl in final_cell_positions[cell_type]:
			matrix_reframe = np.concatenate((matrix_reframe, subl), axis=0)
		final_cell_positions[cell_type] = matrix_reframe[1::]

	def partitionLayer(self, nSublayers):
		# See the wiki page `Placement > Sublayer partitioning` for a detailed explanation
		# of the following steps.
		# TODO: Add to wiki.
		layerThickness = self.layerObject.thickness
		sublayerHeight = layerThickness / nSublayers
		sublayerYs = np.linspace(sublayerHeight, layerThickness, nSublayers)
		sublayerYs = np.insert(sublayerYs, 0, 0)
		sublayerPartitions = np.column_stack([sublayerYs, np.roll(sublayerYs, -1)])[:-1]
		return sublayerPartitions
