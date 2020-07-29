import cProfile
import numpy as np
import os, sys, pstats
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bsb import Scaffold
from bsb.config import JSONConfig

config = JSONConfig("../test.json")
instance = Scaffold(config)
for i in range(1, 40):
    config.resize(100 + i * 20, 100 + i * 20)
    instance.reset_network_cache()
    cProfile.run("instance.compile_network()", "compile_stats")
    p = pstats.Stats("compile_stats")
    p.strip_dirs().sort_stats("cumulative").print_stats("place", 25)
    print("square size:", config.X)
    sleep(1)
