import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.spatial import distance

def compute_circle(center, radius, n_samples=50):
	''' Create circle n_samples starting from given
	center and radius.'''
	nodes = np.linspace(0,2*np.pi,n_samples, endpoint=False)
	x, y = np.sin(nodes)*radius+center[0], np.cos(nodes)*radius+center[1]
	return np.column_stack([x,y])

def linear_project(center, cell, eps):
	''' Linear projections of points on a circle;
	center: center of circle
	cell: radius
	eps: random positive number '''
	return (cell-(center-cell)) + (np.sign(cell-center)*eps)

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
