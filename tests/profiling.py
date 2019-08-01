import cProfile
import numpy as np
import os, sys, pstats
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scaffold.scaffold import Scaffold
from scaffold.config import ScaffoldIniConfig

config = ScaffoldIniConfig('../test.ini')
instance = Scaffold(config)
for i in range(1,40):
    config.resize(100 + i * 20, 100 + i * 20)
    instance.resetNetworkCache()
    cProfile.run('instance.compileNetworkArchitecture()', 'compile_stats')
    p = pstats.Stats('compile_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats('place', 25)
    print('square size:', config.X)
    sleep(1)
