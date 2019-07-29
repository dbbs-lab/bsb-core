import cProfile
import numpy as np
import os, sys, pstats
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scaffold.scaffold import Scaffold
from scaffold.config import ScaffoldIniConfig

config = ScaffoldIniConfig('../test.ini')
instance = Scaffold(config)

cProfile.run('instance.compileNetworkArchitecture()', 'compile_stats')
p = pstats.Stats('compile_stats')
p.strip_dirs().sort_stats('cumulative').print_stats(25)
