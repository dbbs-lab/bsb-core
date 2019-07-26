import abc
from .helpers import ConfigurableClass
from .functions import (
	compute_circle,
	define_bounds,
	rec_intersection,
	get_candidate_points
)
from .quantities import parseToRadian
from pprint import pprint
import numpy as np
import math
import random
from scipy.spatial import distance

class PlacementStrategy(ConfigurableClass):
	@abc.abstractmethod
	def place(self, scaffold, cell_type):
		pass

	def get_placement_count(self, cell_type):
		'''
			Get the placement count, assuming that it is proportional to the available volume times the density.
			If it is not, overload this function in your derived class to attain correct placement counts.
		'''

		scaffold = self.scaffold
		layer = self.layer_instance
		available_volume = layer.available_volume
		if not cell_type.ratio is None:
			# Get the placement count of the ratio cell type and multiply their count by the ratio.
			ratioCellType = scaffold.configuration.cell_types[cell_type.ratioTo]
			return int(ratioCellType.placement.get_placement_count(ratioCellType) * cell_type.ratio)
		if not cell_type.planarDensity is None:
			# Calculate the planar density
			return int(layer.X * layer.Z * cell_type.planarDensity)
		if hasattr(self, 'restriction_factor'):
			# Add a restriction factor to the available volume
			return int(available_volume * self.restriction_factor * cell_type.density)
		# Default: calculate N = V * C
		return int(available_volume * cell_type.density)

class LayeredRandomWalk(PlacementStrategy):
	'''
		Implementation of the placement of cells in sublayers via a self avoiding random walk.
	'''

	casts = {
		'distance_multiplier_min': float,
		'distance_multiplier_max': float
	}

	defaults = {
		'distance_multiplier_min': 0.75,
		'distance_multiplier_max': 1.25
	}

	def validate(self):
		# Check if the layer is given and exists.
		config = self.scaffold.configuration
		if not hasattr(self, 'layer'):
			raise Exception("Required attribute Layer missing from {}".format(self.name))
		if not self.layer in config.layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
		self.layer_instance = self.scaffold.configuration.layers[self.layer]
		try:
			if hasattr(self, 'y_restrict'):
				minmax = self.y_restrict.split(',')
				self.restriction_minimum = float(minmax[0])
				self.restriction_maximum = float(minmax[1])
			else:
				self.restriction_minimum = 0.
				self.restriction_maximum = 1.
			self.restriction_factor = self.restriction_maximum - self.restriction_minimum
		except Exception as e:
			raise Exception("Could not parse Y_Restrict attribute '{}' of {}".format(self.y_restrict, self.layer))

	def place(self, cell_type):
		'''
			Cell placement.

			1. Sublayer partitioning:
				Divide the layer into Y-stacked sublayers dependent on the volume still available in the
				layer, the cell radius and the cell density.
			2. Cell placement:

		'''
		# Variables
		scaffold = self.scaffold
		config = scaffold.configuration
		layer = self.layer_instance
		layer_thickness = self.get_restricted_thickness()
		# Virtual layer origin point that applies the Y-Restriction used for example by basket and stellate cells.
		restricted_origin = np.array([
			layer.origin[0],
			layer.origin[1] + layer.thickness * self.restriction_minimum,
			layer.origin[2]
		])
		# Virtual layer dimensions that apply the Y-Restriction used for example by basket and stellate cells.
		restricted_dimensions = np.array([
			layer.dimensions[0],
			layer_thickness,
			layer.dimensions[2]
		])
		cell_radius = cell_type.radius
		cell_bounds = np.column_stack((
			restricted_origin + cell_radius,
			restricted_origin + restricted_dimensions - cell_radius
		))
		# Get the number of cells that belong in the available volume.
		n_cells_to_place = self.get_placement_count(cell_type)
		if n_cells_to_place == 0:
			print("[WARNING] Volume or density too low, no '{}' cells will be placed".format(cell_type.name))
			n_sublayers = 1
			cell_type.ϵ = 0.
		else:
			# Calculate the volume available per cell
			cell_type.placement_volume = layer.available_volume * self.restriction_factor / n_cells_to_place
			# Calculate the radius of that volume's sphere
			cell_type.placement_radius = (0.75 * cell_type.placement_volume / np.pi) ** (1. / 3.0)
			# Calculate the cell epsilon: This is the length of the 'spare space' a cell has inside of its volume
			cell_type.ϵ = cell_type.placement_radius - cell_radius
			# Calculate the amount of sublayers
			n_sublayers = np.round(layer_thickness / (1.5 * cell_type.placement_radius))
		## Sublayer partitioning
		partitions = self.partition_layer(n_sublayers)
		# Adjust partitions for cell radius.
		partitions = partitions + np.array([cell_radius, -cell_radius])

		## Placement
		min_ϵ = self.distance_multiplier_min * cell_type.ϵ
		max_ϵ = self.distance_multiplier_max * cell_type.ϵ
		cells_per_sublayer = np.round(n_cells_to_place / n_sublayers)

		layer_cell_positions = np.empty((0, 3))

		for sublayer_id in np.arange(n_sublayers):
			if cells_per_sublayer == 0:
				continue
			sublayer_id = int(sublayer_id)
			sublayer_floor = partitions[sublayer_id, 0]
			sublayer_roof = partitions[sublayer_id, 1]

			# Generate the first cell's position.
			starting_position = np.array((
				np.random.uniform(cell_bounds[0, 0], cell_bounds[0, 1]), # X
				np.random.uniform(cell_bounds[1, 0], cell_bounds[1, 1]), # Y
				np.random.uniform(cell_bounds[2, 0], cell_bounds[2, 1]) # Z
			))
			# For Soma and possible points calcs, we take into account only planar coordinates
			center = [starting_position[0], starting_position[2]] # X & Z
			# Get all possible new cell positions
			possible_points = get_candidate_points(center, cell_radius, cell_bounds, min_ϵ, max_ϵ)
			# If there are no possible points, force the cell position to be in the middle of surface
			if possible_points.shape[0] == 0:
				starting_position = np.array([
					cell_bounds[0, 0] + (cell_bounds[0, 1] - cell_bounds[0, 0]) / 2., # X
					np.random.uniform(sublayer_floor, sublayer_roof), # Y
					cell_bounds[2, 0] + (cell_bounds[2, 1] - cell_bounds[2, 0]) / 2. # Z
				])
				possible_points = get_candidate_points(center, cell_radius, cell_bounds, min_ϵ, max_ϵ)
				if possible_points.shape[0] == 0:
					print("[WARNING] Could not place a single cell in {} {} starting from the middle of the simulation volume: Maybe the volume is too low or cell radius/epsilon too big. Sublayer skipped!".format(
						layer.name,
						sublayer_id
					))
					continue
			sublayer_cell_positions = np.array([starting_position])
			# Add third coordinate to all possible points
			possible_points = np.insert(possible_points, 1, np.random.uniform(sublayer_floor, sublayer_roof, possible_points.shape[0]), axis=1)
			# Randomly select one of possible points
			new_point = possible_points[np.random.randint(possible_points.shape[0])]
			# Add the new point to list of cells positions
			sublayer_cell_positions = np.vstack([sublayer_cell_positions, new_point])
			# History of good possible points still available
			good_points_store = [possible_points]
			# History of 'dead-ends' points
			bad_points = []

			last_position = starting_position
			# Place the rest of the cells for the selected sublayer
			for i in np.arange(1, cells_per_sublayer):
				i = int(i)
				# Create soma as a circle:
				# start from the center of previously fixed cell
				center = last_position[[0, 2]]
				possible_points, rnd_ϵ = get_candidate_points(center, cell_radius, cell_bounds, min_ϵ, max_ϵ, return_ϵ=True)
				inter_cell_soma_dist = cell_radius * 2 + rnd_ϵ
				if possible_points.shape[0] == 0:
					print ("Can't place cells because of volume boundaries")
					break
				# For each candidate, calculate distance from the centers of all the (already placed) cells
				# This comparison is performed ONLY along planar coordinates
				distance_from_centers = distance.cdist(possible_points, sublayer_cell_positions[:,[0,2]])
				# Associate a third dimension
				full_coords = np.insert(possible_points, 1, np.random.uniform(sublayer_floor, sublayer_roof, possible_points.shape[0]), axis=1)
				# Check if any of candidate points is placed at acceptable distance from all of the other cells.
				good_idx = list(np.where(np.sum(distance_from_centers.__ge__(inter_cell_soma_dist), axis=1)==distance_from_centers.shape[1])[0])
				if cell_type.name == 'Glomerulus':
					# If the cell type is Glomerulus, we should take into account GoC positions
					inter_glomgoc_dist = cell_radius + config.cell_types['Golgi Cell'].radius
					distance_from_golgi = distance.cdist(full_coords, scaffold.cells_by_type['Golgi Cell'])
					good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
					good_idx = rec_intersection(good_idx, good_from_goc)
				if cell_type.name == 'Granule Cell':
					# If the cell type is Granule, we should take into account GoC and Gloms positions
					inter_grcgoc_dist = cell_radius + config.cell_types['Golgi Cell'].radius
					inter_grcglom_dist = cell_radius + config.cell_types['Glomerulus'].radius
					distance_from_golgi = distance.cdist(full_coords, scaffold.cells_by_type['Golgi Cell'])
					distance_from_gloms = distance.cdist(full_coords, scaffold.cells_by_type['Glomerulus'])
					good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
					good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
					good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
				if len(good_idx) == 0:
					# If we don't find any possible candidate, let's start from the first cell and see
					# if we can find new candidates from previous cells options
					for j in range(len(good_points_store)):
						possible_points = good_points_store[j][:,[0,2]]
						cand_dist = distance.cdist(possible_points, sublayer_cell_positions[:,[0,2]])
						full_coords = good_points_store[j]
						rnd_eps = np.random.uniform(min_ϵ, max_ϵ)
						pprint(rnd_eps)
						inter_cell_soma_dist = cell_radius * 2 + rnd_eps
						good_idx = list(np.where(np.sum(cand_dist.__ge__(inter_cell_soma_dist), axis=1)==cand_dist.shape[1])[0])
						if cell_type.name == 'Glomerulus':
							distance_from_golgi = distance.cdist(full_coords, scaffold.cells_by_type['Golgi Cell'])
							good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
							good_idx = rec_intersection(good_idx, good_from_goc)
						if cell_type.name == 'Granule Cell':
							distance_from_golgi = distance.cdist(full_coords, scaffold.cells_by_type['Golgi Cell'])
							distance_from_gloms = distance.cdist(full_coords, scaffold.cells_by_type['Glomerulus'])
							good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
							good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
							good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
						if len(good_idx) > 0:
							## If there is at least one good candidate, select one randomly
							new_point_idx = random.sample(list(good_idx), 1)[0]
							center = good_points_store[j][new_point_idx]
							# Commented: No need to compute soma of candidate at this point?
							# soma_outer_points = compute_circle(center[[0,2]], cell_radius)
							sublayer_cell_positions = np.vstack([sublayer_cell_positions, center])
							last_position = center
							break
						else:
							bad_points.append(j)
					# If we can't find a good point, the loop must be stopped
					if len(good_idx) == 0:
						print( "Finished after placing {} out of {} cells".format(i, cells_per_sublayer))
						break
				else:
					# If there is at least one good candidate, select one randomly
					new_point_idx = random.sample(list(good_idx), 1)[0]
					new_position = full_coords[new_point_idx]
					sublayer_cell_positions = np.vstack([sublayer_cell_positions, new_position])

					# Keep track of good candidates for each cell
					good_points_store = [good_points_store[i] for i in range(len(good_points_store)) if i not in bad_points]
					good_points_store.append(full_coords[good_idx])
					last_position = new_position
					bad_points = []

			layer_cell_positions = np.concatenate((layer_cell_positions, sublayer_cell_positions))
			scaffold.placement_stats[cell_type.name]['number_of_cells'].append(layer_cell_positions.shape[0])
			print( "{} sublayer number {} out of {} filled".format(cell_type.name, sublayer_id + 1, n_sublayers))
			break

		scaffold.place_cells(cell_type, layer, layer_cell_positions)

	def partition_layer(self, n_sublayers):
		# Allow restricted placement along the Y-axis.
		yMin = self.restriction_minimum
		layer_thickness = self.get_restricted_thickness()
		sublayerHeight = layer_thickness / n_sublayers
		# Divide the Y axis into equal pieces
		sublayerYs = np.linspace(sublayerHeight, layer_thickness, n_sublayers)
		# Add the bottom of the lowest layer and translate all the points by the layer's Y position, keeping the Y restriction into account
		sublayerYs = np.insert(sublayerYs, 0, 0) + self.layer_instance.origin[1] + yMin * self.layer_instance.thickness
		# Create pairs of points on the Y axis corresponding to the bottom and ceiling of each sublayer partition
		sublayerPartitions = np.column_stack([sublayerYs, np.roll(sublayerYs, -1)])[:-1]
		return sublayerPartitions

	def get_restricted_thickness(self):
		return self.layer_instance.thickness * (self.restriction_maximum - self.restriction_minimum)

class ParallelArrayPlacement(PlacementStrategy):
	'''
		Implementation of the placement of cells in parallel arrays.
	'''
	casts = {
		'extension_x': float,
		'extension_z': float,
		'angle': parseToRadian
	}

	defaults = {
		'angle': 0.08726646259971647 # 5 degrees
	}

	required = ['extension_x', 'extension_z', 'angle']

	def validate(self):
		# Check if the layer is given and exists.
		config = self.scaffold.configuration
		if not hasattr(self, 'layer'):
			raise Exception("Required attribute Layer missing from {}".format(self.name))
		if not self.layer in config.layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
		self.layer_instance = self.scaffold.configuration.layers[self.layer]

	def place(self, cell_type):
		'''
			Cell placement: Create a lattice of parallel arrays/lines in the layer's surface.
		'''
		layer = self.layer_instance
		radius = cell_type.radius
		diameter = 2 * radius
		# Extension of a single array in the X dimension
		extensionX = self.extension_x
		spanX = diameter + extensionX
		# Surface area of the plane to place the cells on
		surfaceArea = layer.X * layer.Z
		# Number of cells
		N = self.get_placement_count(cell_type)
		# Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
		# They are placed in straight lines, tilted by a certain angle by adding a shifting value.
		xPositions = np.arange(start=0., stop=layer.X, step=extensionX)[:-1]
		# Amount of parallel arrays of cells
		nArrays = xPositions.shape[0]
		# cells to distribute along the rows
		cellsPerRow = round(N / nArrays)
		# Calculate the position of the cells along the z-axis.
		zPositions, lengthPerCell = np.linspace(start=0., stop=layer.Z - radius, num=cellsPerRow, retstep=True, endpoint=False)
		# Center the cell soma center to the middle of the unit cell
		zPositions += radius + lengthPerCell / 2
		# Add a random shift to the starting points of the arrays for variation.
		startOffset = np.random.rand() * extensionX
		# The length of the X axis where cells can be placed in.
		boundedX = layer.X - radius * 2
		# The length of the X axis rounded up to a multiple of the unit cell size.
		latticeX = nArrays * extensionX
		# Error introduced in the lattice when it is broken by the modulus.
		latticeError = latticeX - boundedX
		# Epsilon: jitter along the z-axis
		ϵ = self.extension_z / 2

		# See the Wiki `Placement > Purkinje placement` for detailed explanations of the following stepd
		for i in np.arange(zPositions.shape[0]):
			# Shift the arrays at an angle
			angleShift = zPositions[i] * math.tan(self.angle)
			# Apply shift and offset
			x = (xPositions + angleShift + startOffset)
			# Place the cells in a bounded lattice with a little modulus magic
			x = layer.origin[0] + x % boundedX - np.floor(x / boundedX) * latticeError + radius
			# Place them at a uniformly random height throughout the layer.
			y = layer.origin[1] + np.random.uniform(radius, layer.Y - radius, x.shape[0])
			# Place the cells in their z-position with slight jitter
			z = layer.origin[2] + np.array([zPositions[i] + ϵ * (np.random.rand() - 0.5) for _ in np.arange(x.shape[0])])
			# Store this stack's cells
			self.scaffold.place_cells(cell_type, layer, np.column_stack([x, y, z]))
