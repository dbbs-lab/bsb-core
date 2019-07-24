import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import distance
from pprint import pprint

def compute_circle(center, radius, n_samples=50):
	'''
		Create `n_samples` points on a circle based on given `center` and `radius`.

		:param center: XYZ vector of the circle center
		:type center: array-like
		:param radius: Radius of the circle
		:type radius: scalar value
		:param n_samples: Amount of points on the circle.
		:type n_samples: int
	'''
	nodes = np.linspace(0,2*np.pi,n_samples, endpoint=False)
	x, y = np.sin(nodes)*radius+center[0], np.cos(nodes)*radius+center[1]
	return np.column_stack([x,y])

def rec_intersection(*args):
	''' Intersection of 2 or more arrays (using recursion)'''
	if len(args) == 2:
		return np.intersect1d(args[0], args[1])
	return rec_intersection(np.intersect1d(args[0], args[1]), args[2::])

def define_bounds(possible_points, cell_bounds):
	'''
		Compare a 2xN matrix of XZ coordinates to a matrix 2x3 with a minimum column and maximum column of XYZ coordinates.
	'''
	x_mask = (possible_points[:,0].__ge__(cell_bounds[0,0])) & (possible_points[:,0].__le__(cell_bounds[0,1]))
	z_mask = (possible_points[:,1].__ge__(cell_bounds[2,0])) & (possible_points[:,1].__le__(cell_bounds[2,1]))
	return x_mask, z_mask

def get_candidate_points(center, radius, bounds, min_ϵ, max_ϵ):
	# Get n points 2 + rnd_ϵ radii away from the center, see `Wiki > Placement > Layered > Epsilon`
	# TODO: Add wiki doc
	rnd_ϵ = np.random.uniform(min_ϵ, max_ϵ)
	possible_points = compute_circle(center, radius * 2 + rnd_ϵ)
	x_mask, z_mask = define_bounds(possible_points, bounds)
	return possible_points[x_mask & z_mask], rnd_ϵ
