import cProfile
import numpy as np
from profiling.linear_project import test_lp
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scaffold.functions import linear_project as updated
from scaffold_functions import linear_project as original

iters = 100000
random_centers = np.random.rand(iters, 2)
random_outer = np.random.rand(iters, 2)
random_eps = np.random.rand(iters)

print('Original linear_project function:')
cProfile.run('test_lp(random_centers, random_outer, random_eps, original)')

print('New linear_project function:')
cProfile.run('test_lp(random_centers, random_outer, random_eps, updated)')
