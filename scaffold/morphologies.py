import abc, numpy as np
from .helpers import ConfigurableClass

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
