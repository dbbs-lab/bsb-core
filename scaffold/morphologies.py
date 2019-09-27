import abc, numpy as np, pickle, h5py
from .helpers import ConfigurableClass
from .output import TreeHandler
from .voxels import VoxelCloud

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

	def init_voxel_cloud(self, voxel_data, voxel_meta, voxel_map):
		'''
			Initialize this Morphology with a voxel cloud from a MorphologyRepository.
		'''
		# Initialise as a true morphology
		self.cloud = VoxelCloud(voxel_meta['grid_size'], voxel_data, voxel_map)

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

class MorphologyRepository(HDF5TreeHandler):

    defaults = {
        'file': 'morphology_repository.hdf5'
    }

    def __init__(self, file=None):
        super().__init__()
        self.handle = None
        if not file is None:
            self.file = file

	# Abstract function from ResourceHandler
    def get_handle(self, mode='r+'):
        '''
            Open the HDF5 storage resource and initialise the MorphologyRepository structure.
        '''
        # Open a new handle to the HDF5 resource.
        handle = HDF5ResourceHandler.get_handle(self, mode)
        # Repository structure missing from resource? Create it.
        if not 'morphologies' in handle:
            handle.create_group('morphologies')
        if not 'morphologies/voxel_clouds' in handle:
            handle.create_group('morphologies/voxel_clouds')
        # Return the handle to the resource.
        return handle

    def import_swc(self, file, name, tags=[], overwrite=False):
        '''
            Import and store .swc file contents as a morphology in the repository.
        '''
        # Read as CSV
        swc_data = np.loadtxt(file)
        # Create empty dataset
        dataset_length = len(swc_data)
        dataset_data = np.empty((dataset_length, 10))
        # Map parent id's to start coordinates. Root node (id: -1) is at 0., 0., 0.
        starts = {-1: [0., 0., 0.]}
        # Iterate over the compartments
        for i in range(dataset_length):
            # Extract compartment record
            compartment = swc_data[i, :]
            compartment_id = compartment[0]
            compartment_type = compartment[1]
            compartment_parent = compartment[6]
            # Check if parent id is known
            if not compartment_parent in starts:
                raise Exception("Node {} references a parent node {} that isn't know yet".format(compartment_id, compartment_parent))
            # Use parent endpoint as startpoint, get endpoint and store it as a startpoint for child compartments
            compartment_start = starts[compartment_parent]
            compartment_end = compartment[2:5]
            starts[compartment_id] = compartment_end
            # Get more compartment radius
            compartment_radius = compartment[5]
            # Store compartment in the repository dataset
            dataset_data[i] = [
                compartment_id,
                compartment_type,
                *compartment_start,
                *compartment_end,
                compartment_radius,
                compartment_parent
            ]
        # Save the dataset in the repository
        with self.load() as repo:
            if overwrite: # Do we overwrite previously existing dataset with same name?
                self.remove_morphology(name) # Delete anything that might be under this name.
            elif self.morphology_exists(name):
                raise Exception("A morphology called '{}' already exists in this repository.")
            # Create the dataset
            dset = repo['morphologies'].create_dataset(name, data=dataset_data)
            # Set attributes
            dset.attrs['name'] = name
            dset.attrs['type'] = 'swc'

    def get_morphology(self, name):
        '''
            Load a morphology from repository data
        '''
        # Open repository and close afterwards
        with self.load() as repo:
            # Check if morphology exists
            if not self.morphology_exists(name):
                raise Exception("Attempting to load unknown morphology '{}'".format(name))
            # Take out all the data with () index, and send along the metadata stored in the attributes
            data = self.raw_morphology(name)
            repo_data = data[()]
            repo_meta = dict(data.attrs)
            voxel_kwargs = {}
            if self.voxel_cloud_exists(name):
                voxels = self.raw_voxel_cloud(name)
                voxel_kwargs['voxel_data'] = voxels['positions'][()]
                voxel_kwargs['voxel_meta'] = dict(voxels.attrs)
                voxel_kwargs['voxel_map'] = pickle.loads(voxels['map'][()])
            return Morphology.from_repo_data(repo_data, repo_meta, **voxel_kwargs)

    def morphology_exists(self, name):
        with self.load() as repo:
            return name in self.handle['morphologies']

    def voxel_cloud_exists(self, name):
        with self.load() as repo:
            return name in self.handle['morphologies/voxel_clouds']

    def remove_morphology(self, name):
        with self.load() as repo:
            if self.morphology_exists(name):
                del self.handle['morphologies/' + name]

    def remove_voxel_cloud(self, name):
        with self.load() as repo:
            if self.voxel_cloud_exists(name):
                del self.handle['morphologies/voxel_clouds/' + name]

    def list_all_morphologies(self):
        with self.load() as repo:
            return list(filter(lambda x: x != 'voxel_clouds', repo['morphologies'].keys()))

    def list_all_voxelized(self):
        with self.load() as repo:
            all = list(repo['morphologies'].keys())
            voxelized = list(filter(lambda x: x in repo['/morphologies/voxel_clouds'], all))
            return voxelized

    def raw_morphology(self, name):
        '''
            Return the morphology dataset
        '''
        with self.load() as repo:
        	return repo['morphologies/' + name]

    def raw_voxel_cloud(self, name):
        '''
            Return the morphology dataset
        '''
        with self.load() as repo:
            return repo['morphologies/voxel_clouds/' + name]
