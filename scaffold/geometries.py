import abc
from .helpers import ConfigurableClass


class Geometry(ConfigurableClass):

	def __init__(self):
		pass

class GranuleCellGeometry(Geometry):
	casts = {
		'dendrite_length' : float,
		'pf_height': float,
		'pf_height_sd': float,
	}
	required = ['dendrite_length', 'pf_height', 'pf_height_sd']

	def validate(self):
		pass

class PurkinjeCellGeometry(Geometry):
	def validate(self):
		pass

class GolgiCellGeometry(Geometry):
	casts = {
		'dendrite_radius': float,
		'axon_x': float,
		'axon_y': float,
		'axon_z': float,
	}

	required = ['dendrite_radius']

	def validate(self):
		pass

class RadialGeometry(Geometry):
	casts = {
		'radius': float,
	}

	required = ['radius']
	
	def validate(self):
		pass

class NoGeometry(Geometry):
	def validate(self):
		pass
