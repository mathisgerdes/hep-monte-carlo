from inspect import getsourcefile
from os import path
import sys
import numpy as np

current_dir = path.dirname(path.abspath(getsourcefile(lambda: 0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
import monte_carlo as mc
sys.path.pop(0)


def get_sampler(ndim, initial_value):
    target = mc.densities.Camel(ndim)
    initial = np.full(ndim, initial_value)
    return mc.DefaultMetropolis(ndim, target), initial
