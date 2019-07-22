import abc
from .helpers import ConfigurableClass
from .functions import compute_circle, define_bounds, rec_intersection, linear_project
from pprint import pprint
import numpy as np
import random
from scipy.spatial import distance

class PlacementStrategy(ConfigurableClass):
	@abc.abstractmethod
	def place(self, scaffold, cellType):
		pass

	def getPlacementCount(self, cellType):
		'''
			Get the placement count, assuming that it is proportional to the available volume times the density.
			If it is not, overload this function in your derived class to attain correct placement counts.
		'''

		scaffold = self.scaffold
		layer = self.layerObject
		availableVolume = layer.availableVolume
		if not cellType.ratio is None:
			# Get the placement count of the ratio cell type and multiply their count by the ratio.
			ratioCellType = scaffold.configuration.CellTypes[cellType.ratioTo]
			return int(ratioCellType.placement.getPlacementCount(ratioCellType) * cellType.ratio)
		if not cellType.planarDensity is None:
			# Calculate the planar density
			return int(self.scaffold.X * self.scaffold.Z * cellType.planarDensity)
		if hasattr(self, 'restrictionFactor'):
			# Add a restriction factor to the available volume
			return int(availableVolume * self.restrictionFactor * cellType.density)
		# Default: calculate N = V * C
		return int(availableVolume * cellType.density)

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

	def validate(self):
		# Check if the layer is given and exists.
		config = self.scaffold.configuration
		if not hasattr(self, 'layer'):
			raise Exception("Required attribute Layer missing from {}".format(self.name))
		if not self.layer in config.Layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
		self.layerObject = self.scaffold.configuration.Layers[self.layer]
		try:
			if hasattr(self, 'y_restrict'):
				minmax = self.y_restrict.split(',')
				self.restrictionMinimum = float(minmax[0])
				self.restrictionMaximum = float(minmax[1])
			else:
				self.restrictionMinimum = 0.
				self.restrictionMaximum = 1.
			self.restrictionFactor = self.restrictionMaximum - self.restrictionMinimum
		except Exception as e:
			raise Exception("Could not parse Y_Restrict attribute '{}' of {}".format(self.y_restrict, self.layer))

	def place(self, cellType):
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
		layer = self.layerObject
		layerThickness = self.getRestrictedThickness()
		# Virtual layer origin point that applies the Y-Restriction used for example by basket and stellate cells.
		restrictedOrigin = np.array([
			layer.origin[0],
			layer.origin[1] + layer.thickness * self.restrictionMinimum,
			layer.origin[2]
		])
		# Virtual layer dimensions that apply the Y-Restriction used for example by basket and stellate cells.
		restrictedDimensions = np.array([
			layer.dimensions[0],
			layerThickness,
			layer.dimensions[2]
		])
		cellRadius = cellType.radius
		cellBounds = np.column_stack((
			restrictedOrigin + cellRadius,
			restrictedOrigin + restrictedDimensions - cellRadius
		))
		# Get the number of cells that belong in the available volume.
		nCellsToPlace = self.getPlacementCount(cellType)
		if nCellsToPlace == 0:
			print("[WARNING] Volume or density too low, no '{}' cells will be placed".format(cellType.name))
			nSublayers = 1
			cellType.ϵ = 0.
		else:
			# Calculate the volume available per cell
			cellType.placementVolume = layer.availableVolume * self.restrictionFactor / nCellsToPlace
			# Calculate the radius of that volume's sphere
			cellType.placementRadius = (0.75 * cellType.placementVolume / np.pi) ** (1. / 3.0)
			# Calculate the cell epsilon: This is the length of the 'spare space' a cell has inside of its volume
			cellType.ϵ = cellType.placementRadius - cellRadius
			# Calculate the amount of sublayers
			nSublayers = np.round(layerThickness / (1.5 * cellType.placementRadius))
		## Sublayer partitioning
		partitions = self.partitionLayer(nSublayers)
		# Adjust partitions for cell radius.
		partitions = partitions + np.array([cellRadius, -cellRadius])

		## Placement
		min_mult = self.distance_multiplier_min
		max_mult = self.distance_multiplier_max
		cellsPerSublayer = np.round(nCellsToPlace / nSublayers)

		layerCellPositions = np.empty((0, 3))

		for sublayerId in np.arange(nSublayers):
			if cellsPerSublayer == 0:
				continue
			sublayerId = int(sublayerId)
			sublayerFloor = partitions[sublayerId, 0]
			sublayerRoof = partitions[sublayerId, 1]
			# Generate the first cell's position.
			startingPosition = np.array((
				np.random.uniform(cellBounds[0, 0], cellBounds[0, 1]), # X
				np.random.uniform(cellBounds[1, 0], cellBounds[1, 1]), # Y
				np.random.uniform(cellBounds[2, 0], cellBounds[2, 1]) # Z
			))

			# Store the starting position in the output array. NB: should add a check to
			## verify that the randomly selected position is not occupied by a different cell type
			sublayerCellPositions = np.array([startingPosition])
			# For Soma and possible points calcs, we take into account only planar coordinates
			center = [startingPosition[0], startingPosition[2]] # First and Third columns
			# Compute cell soma limits
			cell_soma = compute_circle(center, cellRadius)
			# Define random (bounded) epsilon value for minimal distance between two cells
			rnd_eps = np.tile(np.random.uniform(cellType.ϵ * (min_mult/4), cellType.ϵ *(max_mult/4)),2)
			# Create new possible centers for next cell
			possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
			# Constrain their possible positions considering volume (plane) boundaries
			x_mask, z_mask = define_bounds(possible_points, cellBounds)
			possible_points = possible_points[x_mask & z_mask]
			# If there are no possible points, force the cell position to be in the middle of surface
			if possible_points.shape[0] == 0:
				startingPosition = np.array([
					cellBounds[0, 0] + (cellBounds[0, 1] - cellBounds[0, 0]) / 2., # X
					np.random.uniform(sublayerFloor, sublayerRoof), # Y
					cellBounds[2, 0] + (cellBounds[2, 1] - cellBounds[2, 0]) / 2. # Z
				])
				cell_soma = compute_circle(center, cellRadius)
				possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
				x_mask, z_mask = define_bounds(possible_points, cellBounds)
				possible_points = possible_points[x_mask & z_mask]
				if possible_points.shape[0] == 0:
					print("[WARNING] Could not place a single cell in {} {} starting from the middle of the simulation volume: Maybe the volume is too low or cell radius/epsilon too big. Sublayer skipped!".format(
						layer.name,
						sublayerId
					))
					continue
			# Add third coordinate to all possible points
			possible_points = np.insert(possible_points, 1, np.random.uniform(sublayerFloor, sublayerRoof, possible_points.shape[0]), axis=1)
			# Randomly select one of possible points
			new_point = possible_points[np.random.randint(possible_points.shape[0])]
			# Add the new point to list of cells positions
			sublayerCellPositions = np.vstack([sublayerCellPositions, new_point])
			# History of good possible points still available
			good_points_store = [possible_points]
			# History of 'dead-ends' points
			bad_points = []

			# Place the rest of the cells for the selected sublayer
			for i in np.arange(1, cellsPerSublayer):
				i = int(i)
				# Create soma as a circle:
				# start from the center of previously fixed cell
				center = sublayerCellPositions[-1][[0,2]]
				# Sample n_samples points along the circle surrounding cell center
				cell_soma = compute_circle(center, cellRadius)
				# Recalc random epsilon and use it to define min distance from current cell
				# for possible new cells
				rnd_eps = np.tile(np.random.uniform(cellType.ϵ*(min_mult/4), cellType.ϵ*(max_mult/4)),2)
				inter_cell_soma_dist = cellRadius*2+rnd_eps[0]
				possible_points = np.array([linear_project(center, cell, rnd_eps) for cell in cell_soma])
				x_mask, z_mask = define_bounds(possible_points, cellBounds)
				possible_points = possible_points[x_mask & z_mask]
				if possible_points.shape[0] == 0:
					print ("Can't place cells because of volume boundaries")
					break
				# For each candidate, calculate distance from the centers of all the (already placed) cells
				# This comparison is performed ONLY along planar coordinates
				distance_from_centers = distance.cdist(possible_points, sublayerCellPositions[:,[0,2]])
				# Associate a third dimension
				full_coords = np.insert(possible_points, 1, np.random.uniform(sublayerFloor, sublayerRoof, possible_points.shape[0]), axis=1)
				# Check if any of candidate points is placed at acceptable distance from all of the other cells.
				good_idx = list(np.where(np.sum(distance_from_centers.__ge__(inter_cell_soma_dist), axis=1)==distance_from_centers.shape[1])[0])
				if cellType.name == 'Glomerulus':
					# If the cell type is Glomerulus, we should take into account GoC positions
					inter_glomgoc_dist = cellRadius + config.CellTypes['Golgi Cell'].radius
					distance_from_golgi = distance.cdist(full_coords, scaffold.CellsByType['Golgi Cell'])
					good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
					good_idx = rec_intersection(good_idx, good_from_goc)
				if cellType.name == 'Granule Cell':
					# If the cell type is Granule, we should take into account GoC and Gloms positions
					inter_grcgoc_dist = cellRadius + config.CellTypes['Golgi Cell'].radius
					inter_grcglom_dist = cellRadius + config.CellTypes['Glomerulus'].radius
					distance_from_golgi = distance.cdist(full_coords, scaffold.CellsByType['Golgi Cell'])
					distance_from_gloms = distance.cdist(full_coords, scaffold.CellsByType['Glomerulus'])
					good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
					good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
					good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
				if len(good_idx) == 0:
					# If we don't find any possible candidate, let's start from the first cell and see
					# if we can find new candidates from previous cells options
					for j in range(len(good_points_store)):
						possible_points = good_points_store[j][:,[0,2]]
						cand_dist = distance.cdist(possible_points, sublayerCellPositions[:,[0,2]])
						full_coords = good_points_store[j]
						rnd_eps = np.tile(np.random.uniform(cellType.ϵ * (min_mult/4), cellType.ϵ * (max_mult/4)),2)
						inter_cell_soma_dist = cellRadius*2+rnd_eps[0]
						good_idx = list(np.where(np.sum(cand_dist.__ge__(inter_cell_soma_dist), axis=1)==cand_dist.shape[1])[0])
						if cellType.name == 'Glomerulus':
							distance_from_golgi = distance.cdist(full_coords, scaffold.CellsByType['Golgi Cell'])
							good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_glomgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
							good_idx = rec_intersection(good_idx, good_from_goc)
						if cellType.name == 'Granule Cell':
							distance_from_golgi = distance.cdist(full_coords, scaffold.CellsByType['Golgi Cell'])
							distance_from_gloms = distance.cdist(full_coords, scaffold.CellsByType['Glomerulus'])
							good_from_goc = list(np.where(np.sum(distance_from_golgi.__ge__(inter_grcgoc_dist), axis=1)==distance_from_golgi.shape[1])[0])
							good_from_gloms = list(np.where(np.sum(distance_from_gloms.__ge__(inter_grcglom_dist), axis=1)==distance_from_gloms.shape[1])[0])
							good_idx = rec_intersection(good_idx, good_from_goc, good_from_gloms)
						if len(good_idx) > 0:
							## If there is at least one good candidate, select one randomly
							new_point_idx = random.sample(list(good_idx), 1)[0]
							center = good_points_store[j][new_point_idx]
							cell_soma = compute_circle(center[[0,2]], cellRadius)
							sublayerCellPositions = np.vstack([sublayerCellPositions, center])
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
					sublayerCellPositions = np.vstack([sublayerCellPositions, full_coords[new_point_idx]])

					# Keep track of good candidates for each cell
					good_points_store = [good_points_store[i] for i in range(len(good_points_store)) if i not in bad_points]
					good_points_store.append(full_coords[good_idx])
					bad_points = []

			layerCellPositions = np.concatenate((layerCellPositions, sublayerCellPositions))
			scaffold.placement_stats[cellType.name]['number_of_cells'].append(layerCellPositions.shape[0])
			print( "{} sublayer number {} out of {} filled".format(cellType.name, sublayerId + 1, nSublayers))

		scaffold.placeCells(cellType, layer, layerCellPositions)

	def partitionLayer(self, nSublayers):
		# Allow restricted placement along the Y-axis.
		yMin = self.restrictionMinimum
		layerThickness = self.getRestrictedThickness()
		sublayerHeight = layerThickness / nSublayers
		# Divide the Y axis into equal pieces
		sublayerYs = np.linspace(sublayerHeight, layerThickness, nSublayers)
		# Add the bottom of the lowest layer and translate all the points by the layer's Y position, keeping the Y restriction into account
		sublayerYs = np.insert(sublayerYs, 0, 0) + self.layerObject.origin[1] + yMin * self.layerObject.thickness
		# Create pairs of points on the Y axis corresponding to the bottom and ceiling of each sublayer partition
		sublayerPartitions = np.column_stack([sublayerYs, np.roll(sublayerYs, -1)])[:-1]
		return sublayerPartitions

	def getRestrictedThickness(self):
		return self.layerObject.thickness * (self.restrictionMaximum - self.restrictionMinimum)

class ParallelArrayPlacement(PlacementStrategy):
	'''
		Implementation of the placement of cells in parallel arrays.
	'''
	casts = {
		'extension_x': float,
		'extension_z': float,
	}

	defaults = {

	}

	required = ['extension_x', 'extension_z']

	def validate(self):
		# Check if the layer is given and exists.
		config = self.scaffold.configuration
		if not hasattr(self, 'layer'):
			raise Exception("Required attribute Layer missing from {}".format(self.name))
		if not self.layer in config.Layers:
			raise Exception("Unknown layer '{}' in {}".format(self.layer, self.name))
		self.layerObject = self.scaffold.configuration.Layers[self.layer]

	def place(self, cellType):
		'''
			Cell placement: Create a single layer of multiple arrays of cells parallel to each other.
		'''
		layer = self.layerObject
		radius = cellType.radius
		cellDiameter = 2 * radius
		# Extension of a single array in the X dimension
		extensionX = self.extension_x
		spanX = cellDiameter + extensionX
		# Volume dimensions
		volumeX = cellType.scaffold.configuration.X
		volumeZ = cellType.scaffold.configuration.Z
		# Surface area of the plane to place the cells on
		surfaceArea = volumeX * volumeZ
		# Number of cells
		N = self.getPlacementCount(cellType)
		# Epsilon. # TODO: Better comment description
		ϵ = (( spanX ** 2 - 4. * (cellDiameter * extensionX - (surfaceArea / N))) ** .5 - spanX) / 2.
		# Calculate the z values for each parallel array
		parallelArrayZ = np.linspace(radius, volumeZ - radius - (ϵ / 2), volumeZ / (cellDiameter + ϵ))

		offset = 0
		delta = parallelArrayZ.shape[0] / ((extensionX / 2) - 1)
		npc = 0

		for i in np.arange(parallelArrayZ.shape[0]):

			# Why extension x - 1?
			x = np.arange((extensionX / 2.) + offset + radius, volumeX - radius, extensionX - 1)
			y = np.random.uniform(radius + layer.origin[1], layer.thickness - radius + layer.origin[1], x.shape[0])
			z = np.zeros((x.shape[0]))
			for cont in np.arange(x.shape[0]):
				z[cont] = parallelArrayZ[i] + (ϵ / 2) * np.random.rand()

			self.scaffold.placeCells(cellType, layer, np.column_stack([x, y, z]))
			offset += delta * 5. * np.random.rand()
			npc += x.shape[0]
