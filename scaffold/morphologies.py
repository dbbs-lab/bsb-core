import abc, numpy as np, pickle, h5py
from .helpers import ConfigurableClass
from .voxels import VoxelCloud, detect_box_compartments, Box
from sklearn.neighbors import KDTree

class Compartment:
	def __init__(self, repo_record):
		'''
			Create a compartment from repository data.
		'''
		# Transfer basic data to properties.
		self.id = repo_record[0]
		self.type = repo_record[1]
		self.start = repo_record[2:5]
		self.end = repo_record[5:8]
		self.radius = repo_record[8]
		self.parent = repo_record[9]
		# Calculate midpoint of the compartment
		self.midpoint = (self.end - self.start) / 2 + self.start
		# Calculate the radius of the outer sphere of this compartment
		self.spherical = np.sqrt((self.start[:] - self.end[:]) ** 2) / 2

class Morphology(ConfigurableClass):

	def __init__(self):
		super().__init__()
		self.compartments = None
		self.cloud = None
		self.has_morphology = False
		self.has_voxels = False

	def init_morphology(self, repo_data, repo_meta):
		'''
			Initialize this Morphology with detailed morphology data from a MorphologyRepository.
		'''
		# Initialise as a true morphology
		self.compartments = []
		self.morphology_name = repo_meta['name']
		self.has_morphology = True
		# Iterate over the data to create compartment objects
		for i in range(len(repo_data)):
			repo_record = repo_data[i, :]
			compartment = Compartment(repo_record)
			self.compartments.append(compartment)
		# Create a tree from the compartment object list
		self.compartment_tree = None
		morphology_trees = self.scaffold.trees["morphologies"]
		if self.scaffold:
			self.compartment_tree = morphology_trees.get_tree(morphology_name)
		if self.compartment_tree is None:
			self.compartment_tree = KDTree(np.array(list(map(lambda c: c.end, self.compartments))))
			morphology_trees.add_tree(morphology_name, self.compartment_tree)
			morphology_trees.save()

	def init_voxel_cloud(self, voxel_data, voxel_meta, voxel_map):
		'''
			Initialize this Morphology with a voxel cloud from a MorphologyRepository.
		'''
		bounds = voxel_meta['bounds']
		grid_size = voxel_meta['grid_size']
		# Initialise as a true morphology
		self.cloud = VoxelCloud(bounds, voxel_data, grid_size, voxel_map)

	@staticmethod
	def from_repo_data(repo_data, repo_meta, voxel_data=None, voxel_map=None, voxel_meta=None, scaffold = None):
		# Instantiate morphology instance
		m = TrueMorphology()
		if not scaffold is None:
			# Initialise configurable class
			m.initialise(scaffold)
		# Load the morphology data into this morphology instance
		m.init_morphology(repo_data, repo_meta)
		if not voxel_data is None:
			if voxel_map is None or voxel_meta is None:
				raise Exception("If voxel_data is provided, voxel_meta and voxel_map must be provided aswell.")
			m.init_voxel_cloud(voxel_data, voxel_meta, voxel_map)
		return m

class TrueMorphology(Morphology):
	'''
		Used to load morphologies that don't need to be configured/validated.
	'''

	def validate(self):
		pass

	def voxelize(self, N):
		self.cloud = VoxelCloud.create(self, N)

	def get_compartment_map(self, boxes, voxels, box_size):
		tree = self.compartment_tree
		compartment_map = []
		box_positions = np.column_stack(boxes[:, voxels])
		compartments = tree.get_arrays()[0]
		compartments_taken = set([])
		for i in range(box_positions.shape[0]):
			box_origin = box_positions[i, :]
			compartments_in_outer_sphere = detect_box_compartments(tree, box_origin, box_size)
			candidate_positions = compartments[compartments_in_outer_sphere]
			bool_vector = np.ones(compartments_in_outer_sphere.shape, dtype=bool)
			bool_vector &= (candidate_positions[:, 0] >= box_origin[0]) & (candidate_positions[:, 0] <= box_origin[0] + box_size)
			bool_vector &= (candidate_positions[:, 1] >= box_origin[1]) & (candidate_positions[:, 1] <= box_origin[1] + box_size)
			bool_vector &= (candidate_positions[:, 2] >= box_origin[2]) & (candidate_positions[:, 2] <= box_origin[2] + box_size)
			compartments_in_box = set(compartments_in_outer_sphere[bool_vector]) - compartments_taken
			compartments_taken |= set(compartments_in_box)
			compartment_map.append(list(compartments_in_box))
		return compartment_map

	def get_bounding_box(self, centered=True):
		# Use the compartment tree to get a quick array of the compartments positions
		tree = self.compartment_tree
		compartments = tree.get_arrays()[0]
		# Determine the amount of dimensions of the morphology. Let's hope 3 ;)
		n_dimensions = range(compartments.shape[1])
		# Create a bounding box
		outer_box = Box()
		# The outer box dimensions are equal to the maximum distance between compartments in each of n dimensions
		outer_box.dimensions = np.array([np.max(compartments[:, i]) - np.min(compartments[:, i]) for i in n_dimensions])
		# The outer box origin should be in the middle of the outer bounds if 'centered' is True. (So lowermost point + sometimes half of dimensions)
		outer_box.origin = np.array([np.min(compartments[:, i]) + (outer_box.dimensions[i] / 2) * int(centered) for i in n_dimensions])
		return outer_box

	def get_search_radius(self, plane='xyz'):
		pos = np.array(self.compartment_tree.get_arrays()[0])
		dimensions = ['x', 'y', 'z']
		try:
			max_dists = np.max(np.abs(np.array([pos[:,dimensions.index(d)] for d in plane])), axis=1)
		except ValueError as e:
			raise Exception("Unknown dimensions in dimension string '{}'".format(plane))
		print(max_dists)
		return np.sqrt(np.sum(max_dists ** 2))

	def get_compartment_network(self):
		compartments = self.compartments
		node_list = [set([]) for c in compartments]
		# Fix first and last compartments
		node_list[0] = set([1])
		node_list.append(set([]))
		# Add child nodes to their parent's adjacency set
		for node in compartments[1:]:
			node_list[int(node.parent)].add(int(node.id))
		return node_list

	def get_plot_range(self):
		compartments = self.compartment_tree.get_arrays()[0]
		n_dimensions = range(compartments.shape[1])
		mins = np.array([np.min(compartments[:, i]) for i in n_dimensions])
		max = np.max(np.array([np.max(compartments[:, i]) - mins[i] for i in n_dimensions]))
		return list(zip(mins.tolist(), (mins + max).tolist()))

	def get_compartments_tree(self, type=None):
		if not type is None:
			return self.scaffold.trees["morphologies"].get_sub_tree()

class GranuleCellGeometry(Morphology):
	casts = {
		'dendrite_length' : float,
		'pf_height': float,
		'pf_height_sd': float,
	}
	required = ['dendrite_length', 'pf_height', 'pf_height_sd']

	def validate(self):
		pass

class PurkinjeCellGeometry(Morphology):
	def validate(self):
		pass

class GolgiCellGeometry(Morphology):
	casts = {
		'dendrite_radius': float,
		'axon_x': float,
		'axon_y': float,
		'axon_z': float,
	}

	required = ['dendrite_radius']

	def validate(self):
		pass

class RadialGeometry(Morphology):
	casts = {
		'dendrite_radius': float,
	}

	required = ['dendrite_radius']

	def validate(self):
		pass

class NoGeometry(Morphology):
	def validate(self):
		pass
