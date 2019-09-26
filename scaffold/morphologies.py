import abc, numpy as np
from .helpers import ConfigurableClass
from .output import ResourceHandler

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
		self.has_morphology = False

	def load_morphology(self, repo_data, repo_meta):
		'''
			Load a morphology from the repository
		'''
		# Initialise as a true morphology
		self.compartments = []
		self.morphology_name = repo_meta['name']
		self.has_morphology = True
		# Iterate over the data to create compartment objects
		for i in range(len(repo_data)):
			repo_record = repo_data[i, :]
			self.compartments.append(Compartment(repo_record))

	@staticmethod
	def from_repo_data(repo_data, repo_meta, scaffold = None):
		# Instantiate morphology instance
		m = TrueMorphology()
		if not scaffold is None:
			# Initialise configurable class
			m.initialise(scaffold)
		# Load the morphology data into this morphology instance
		m.load_morphology(repo_data, repo_meta)
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

class MorphologyRepository(ResourceHandler):

    defaults = {
        'file': 'morphology_repository.hdf5'
    }

    def __init__(self, file=None):
        super().__init__()
        self.handle = None
        if not file is None:
            self.file = file

    def get_handle(self):
        '''
            Open the MorphologyRepository storage resource.
        '''
        if not self.handle is None: # Resource already open?
            # Return the handle to the already open resource.
            return self.handle
        # Open a new handle to the resource.
        self.handle = h5py.File(self.file)
        # Repository structure missing from resource? Create it.
        if not 'morphologies' in self.handle:
            self.handle.create_group('morphologies')
        if not 'morphologies/voxel_clouds' in self.handle:
            self.handle.create_group('morphologies/voxel_clouds')
        # Return the handle to the resource.
        return self.handle

    def release_handle(self, handle):
        '''
            Close the MorphologyRepository storage resource.
        '''
        self.handle = None
        return handle.close()

    def save(self):
        '''
            Called when the scaffold is saving itself.
            Don't need to do anything special with the repo when the scaffold is saving itself.
        '''
        pass

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
                self._rmm(name) # Delete anything that might be under this name.
            elif self._me(name):
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
            if not self._me(name):
                raise Exception("Attempting to load unknown morphology '{}'".format(name))
            # Take out all the data with () index, and send along the metadata stored in the attributes
            data = self._m(name)
            return Morphology.from_repo_data(data[()], data.attrs)

    def morphology_exists(self, name):
        with self.load() as repo:
            return self._me(name)

    def voxel_cloud_exists(self, name):
        with self.load() as repo:
            return self._ve(name)

    def remove_morphology(self, name):
        with self.load() as repo:
            self._rmm(name)

    def remove_voxel_cloud(self, name):
        with self.load() as repo:
            self._rmv(name)

    def list_all_morphologies(self):
        with self.load() as repo:
            return list(filter(lambda x: x != 'voxel_clouds', repo['morphologies'].keys()))

    def list_all_voxelized(self):
        with self.load() as repo:
            all = list(repo['morphologies'].keys())
            voxelized = list(filter(lambda x: x in repo['/morphologies/voxel_clouds'], all))
            return voxelized

    #-- Handle avoidance shorthand functions
    # These function are shorthands for internal use that assume an open handle
    # in self.handle and don't close that handle.

    def _me(self, name):
        '''
            Shorthand for self.morphology_exists
        '''
        return name in self.handle['morphologies']

    def _ve(self, name):
        '''
            Shorthand for self.voxel_cloud_exists
        '''
        return name in self.handle['morphologies/voxel_clouds']

    def _rmm(self, name):
        '''
            Shorthand for self.remove_morphology
        '''
        if self._me(name):
            del self.handle['morphologies/' + name]

    def _rmv(self, name):
        '''
            Shorthand for self.remove_voxel_cloud
        '''
        if self._ve(name):
            del self.handle['morphologies/voxel_clouds' + name]

    def _m(self, name):
        '''
            Return the morphology dataset
        '''
        return self.handle['morphologies/' + name]
