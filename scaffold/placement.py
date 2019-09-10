import abc, math, random, numpy as np
from .helpers import ConfigurableClass
from scipy.spatial import distance
from .functions import (
	compute_circle,
	define_bounds,
	rec_intersection,
	get_candidate_points,
	add_y_axis,
	exclude_index
)

class PlacementStrategy(ConfigurableClass):
	def __init__(self):
		super().__init__()
		self.layer = None
		self.radius = None
		self.density = None
		self.planar_density = None
		self.placement_count_ratio = None
		self.density_ratio = None
		self.placement_relative_to = None

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
		placement = cell_type.placement
		if not placement.placement_count_ratio is None:
			# Get the placement count of the ratio cell type and multiply their count by the ratio.
			ratioCellType = scaffold.configuration.cell_types[placement.placement_relative_to]
			return int(ratioCellType.placement.get_placement_count(ratioCellType) * placement.placement_count_ratio)
		if not placement.density_ratio is None:
			# Get the density of the ratio cell type and multiply it by the ratio.
			ratioCellType = scaffold.configuration.cell_types[placement.placement_relative_to]
			relation = ratioCellType.placement
			ratio = placement.placement_count_ratio
			n1 = relation.get_placement_count(ratioCellType)
			V1 = relation.layer_instance.volume
			V2 = layer.volume
			return int(n1 * ratio * V2 / V1)
		if not placement.planar_density is None:
			# Calculate the planar density
			return int(layer.width * layer.depth * placement.planar_density)
		if hasattr(self, 'restriction_factor'):
			# Add a restriction factor to the available volume
			return int(available_volume * self.restriction_factor * placement.density)
		# Default: calculate N = V * C
		return int(available_volume * placement.density)

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
			raise Exception("Required attribute 'layer' missing from {}".format(self.name))
		if not self.layer in config.layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
		self.layer_instance = self.scaffold.configuration.layers[self.layer]
		try:
			if hasattr(self, 'y_restriction'):
				self.restriction_minimum = float(self.y_restriction[0])
				self.restriction_maximum = float(self.y_restriction[1])
			else:
				self.restriction_minimum = 0.
				self.restriction_maximum = 1.
			self.restriction_factor = self.restriction_maximum - self.restriction_minimum
		except Exception as e:
			raise Exception("Invalid y_restriction attribute '{}' of {}".format(self.y_restriction, self.layer))

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
		cell_radius = cell_type.placement.radius
		cell_bounds = np.column_stack((
			restricted_origin + cell_radius,
			restricted_origin + restricted_dimensions - cell_radius
		))
		# Get the number of cells that belong in the available volume.
		n_cells_to_place = self.get_placement_count(cell_type)
		if n_cells_to_place == 0:
			if scaffold.configuration.verbosity > 0:
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
		previously_placed_cells = scaffold.cells_by_layer[layer.name][:,[2,3,4]]
		previously_placed_types = np.array(scaffold.cells_by_layer[layer.name][:,1], dtype=int)
		other_cell_type_radii = np.array(list(map(lambda type: scaffold.configuration.cell_types[type].placement.radius, scaffold.configuration.cell_type_map)))
		if len(previously_placed_cells) > 0:
			previously_placed_min_dist = other_cell_type_radii[previously_placed_types] + cell_radius
		else:
			previously_placed_min_dist = np.empty((0))
		other_cell_type_count = previously_placed_min_dist.shape[0]

		for sublayer_id in np.arange(n_sublayers):
			if cells_per_sublayer == 0:
				continue
			sublayer_id = int(sublayer_id)
			sublayer_floor = partitions[sublayer_id, 0]
			sublayer_roof = partitions[sublayer_id, 1]
			sublayer_attempts = 0

			# Generate the first cell's position.
			starting_position = np.array((
				np.random.uniform(cell_bounds[0, 0], cell_bounds[0, 1]), # X
				np.random.uniform(cell_bounds[1, 0], cell_bounds[1, 1]), # Y
				np.random.uniform(cell_bounds[2, 0], cell_bounds[2, 1])  # Z
			))
			planar_start = [starting_position[0], starting_position[2]] # X & Z
			# Get all possible new cell positions
			planar_candidates = get_candidate_points(planar_start, cell_radius, cell_bounds, min_ϵ, max_ϵ)
			# If there are no possible points, force the cell position to be in the middle of surface
			if planar_candidates.shape[0] == 0:
				starting_position = np.array([
					cell_bounds[0, 0] + (cell_bounds[0, 1] - cell_bounds[0, 0]) / 2., # X
					np.random.uniform(sublayer_floor, sublayer_roof), 				  # Y
					cell_bounds[2, 0] + (cell_bounds[2, 1] - cell_bounds[2, 0]) / 2.  # Z
				])
				planar_candidates = get_candidate_points(planar_start, cell_radius, cell_bounds, min_ϵ, max_ϵ)
				if planar_candidates.shape[0] == 0:
					if scaffold.configuration.verbosity > 0:
						print("[WARNING] Could not place a single cell in {} {} starting from the middle of the simulation volume: Maybe the volume is too low or cell radius/epsilon too big. Sublayer skipped!".format(
							layer.name,
							sublayer_id
						))
					continue
			placed_positions = np.array([starting_position])
			planar_placed_positions = np.array([starting_position[[0,2]]])
			full_coords = add_y_axis(planar_placed_positions, sublayer_floor, sublayer_roof)
			good_points_store = [np.copy(full_coords)]
			last_position = starting_position
			for current_cell_count in np.arange(1, cells_per_sublayer, dtype=int):
				# Subtract failed placement attempts from loop counter
				current_cell_count = current_cell_count - sublayer_attempts
				planar_candidates, rnd_ϵ = get_candidate_points(last_position[[0, 2]], cell_radius, cell_bounds, min_ϵ, max_ϵ, return_ϵ=True)
				full_coords = add_y_axis(planar_candidates, sublayer_floor, sublayer_roof)
				inter_cell_soma_dist = cell_radius * 2 + rnd_ϵ
				sublayer_distances = distance.cdist(planar_candidates, planar_placed_positions)
				good_indices = list(np.where(np.sum(sublayer_distances > inter_cell_soma_dist, axis=1) == current_cell_count)[0])
				planar_candidates = planar_candidates[good_indices]
				full_coords = full_coords[good_indices]
				layer_distances = distance.cdist(full_coords, previously_placed_cells)
				good_indices = list(np.where(np.sum(layer_distances > previously_placed_min_dist, axis=1) == other_cell_type_count)[0])
				if len(good_indices) == 0:
					max_attempts = len(good_points_store)
					for attempt in range(max_attempts):
						store_id = np.random.randint(max_attempts - attempt)
						planar_candidates = good_points_store[store_id][:,[0,2]]
						full_coords = good_points_store[store_id]
						rnd_ϵ = np.random.uniform(min_ϵ, max_ϵ)
						inter_cell_soma_dist = cell_radius * 2 + rnd_ϵ
						sublayer_distances = distance.cdist(planar_candidates, planar_placed_positions)
						good_indices = list(np.where(np.sum(sublayer_distances > inter_cell_soma_dist, axis=1)==current_cell_count)[0])
						planar_candidates = planar_candidates[good_indices]
						full_coords = full_coords[good_indices]
						layer_distances = distance.cdist(full_coords, previously_placed_cells)
						good_indices = list(np.where(np.sum(layer_distances > previously_placed_min_dist, axis=1) == other_cell_type_count)[0])
						if len(good_indices) > 0:
							random_index = random.sample(good_indices, 1)[0]
							candidate = full_coords[random_index]
							placed_positions = np.vstack([placed_positions, candidate])
							planar_placed_positions = np.vstack([planar_placed_positions, candidate[[0,2]]])
							last_position = candidate
							break
						else:
							good_points_store = exclude_index(good_points_store, store_id)
					if len(good_indices) == 0:
						if sublayer_attempts < 10:
							sublayer_attempts += 1
							# Try again from a random position
							last_position = np.array([
								cell_bounds[0, 0] + (cell_bounds[0, 1] - cell_bounds[0, 0]) / 2., # X
								np.random.uniform(sublayer_floor, sublayer_roof), 				  # Y
								cell_bounds[2, 0] + (cell_bounds[2, 1] - cell_bounds[2, 0]) / 2.  # Z
							])
						else:
							if scaffold.configuration.verbosity > 2:
								print( "Finished after placing {} out of {} cells".format(current_cell_count, cells_per_sublayer))
							break
				else:
					random_index = random.sample(good_indices, 1)[0]
					new_position = full_coords[random_index]
					placed_positions = np.vstack([placed_positions, new_position])
					planar_placed_positions = np.vstack([planar_placed_positions, new_position[[0,2]]])

					good_points_store.append(full_coords[good_indices])
					last_position = new_position

			layer_cell_positions = np.concatenate((layer_cell_positions, placed_positions))
			scaffold.placement_stats[cell_type.name]['number_of_cells'].append(layer_cell_positions.shape[0])
			if scaffold.configuration.verbosity > 2:
				print( "{} sublayer number {} out of {} filled".format(cell_type.name, sublayer_id + 1, n_sublayers))

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
		'angle': lambda x: float(x) / 360 * math.pi
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
		radius = cell_type.placement.radius
		diameter = 2 * radius
		# Extension of a single array in the X dimension
		extensionX = self.extension_x
		spanX = diameter + extensionX
		# Surface area of the plane to place the cells on
		surfaceArea = layer.width * layer.depth
		# Number of cells
		N = self.get_placement_count(cell_type)
		# Place purkinje cells equally spaced over the entire length of the X axis kept apart by their dendritic trees.
		# They are placed in straight lines, tilted by a certain angle by adding a shifting value.
		xPositions = np.arange(start=0., stop=layer.width, step=extensionX)[:-1]
		# Amount of parallel arrays of cells
		nArrays = max(1, xPositions.shape[0])
		# cells to distribute along the rows
		cellsPerRow = round(N / nArrays)
		# Calculate the position of the cells along the z-axis.
		zPositions, lengthPerCell = np.linspace(start=0., stop=layer.depth - radius, num=cellsPerRow, retstep=True, endpoint=False)
		# Center the cell soma center to the middle of the unit cell
		zPositions += radius + lengthPerCell / 2
		# Add a random shift to the starting points of the arrays for variation.
		startOffset = np.random.rand() * extensionX
		# The length of the X axis where cells can be placed in.
		boundedX = layer.width - radius * 2
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
			y = layer.origin[1] + np.random.uniform(radius, layer.height - radius, x.shape[0])
			# Place the cells in their z-position with slight jitter
			z = layer.origin[2] + np.array([zPositions[i] + ϵ * (np.random.rand() - 0.5) for _ in np.arange(x.shape[0])])
			# Store this stack's cells
			self.scaffold.place_cells(cell_type, layer, np.column_stack([x, y, z]))
