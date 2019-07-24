import numpy as np

def test_lp(centers, outers, eps, cb):
    for i in range(1, len(centers)):
        cb(centers[i], outers[i], eps[i])
