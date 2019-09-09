import abc
from .helpers import ConfigurableClass


class Morphology(ConfigurableClass):

	def __init__(self):
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
