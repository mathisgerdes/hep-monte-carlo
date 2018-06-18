""" Module for Monte Carlo integration methods.

The integral for all techniques is performed over the unit hypercube [0, 1]^ndim.
The integrands must return one dimensional values (e.g. floats)

The module contains several Monte Carlo integration methods, each of which
is wrapped in a class. To create an integrator for certain method, create
an object specifying general settings such as dimensionality or method-specific
parameters. The object is callable with a function (integrand) and iteration
settings to compute the integral. They return the integral estimate and the
(natural estimator of the) standard deviation of that estimate.

Example:
    >>> mc = MonteCarloPlain()  # defaults to one dimensional integration.
    >>> # Integrate fn(x) = x, using 1000 sample points.
    >>> est, err = mc(lambda x: x, 1000)

Advanced variance-reducing Monte Carlo techniques contain several phases,
for each of which the number of iterations and number of function evaluations
need to be specified. Since the structure of the algorithms differ,
the call signature is not the same for all methods.

Example:
    Divide the integration space into 4 equally sized partitions with a base
    number of 10 sample points in each volume.
    >>> volumes = GridVolumes(ndim=1, divisions=4, default_count=10)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> # Stratified sampling expects a multiple instead of a total sample size.
    >>> est, err = mc_strat(lambda x: x, 5)  # 5 * 10 sample points per region

To allow sensible comparisons of the different techniques, they all provide
a function for creating an interface where only the total number of function
evaluations (which are considered to be expensive -- the asymptotic efficiency
is usually in this variable) and the integrand are passed.

Example:
    >>> volumes = GridVolumes(ndim=1, divisions=4, default_count=100)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> mc_strat_ = mc_strat.get_interface_infer_multiple()
    >>> est, err = mc_strat(lambda x: x, 4000)  # multiple is 4000/(4*100) = 40

The integrand passed to a ndim-dimensional method must take ndim arguments, all
of which are 1d numpy arrays of some size N, and return a numpy array of
the same length N. The numpy vectorize function can help if the integrand of
interest cannot be written using vectors (arrays) easily
(note, however, that vectorize does not increase efficiency).

Example:
    >>> def fn(x, y): return np.sin(x + y)  # x and y are floats
    >>> fn = np.vectorize(fn)                # use @np.vectorize instead
    >>> answer = fn([np.pi, 0], [0, np.pi/2])
    >>> np.allclose(np.array([0, 1]), answer)
    True

    Or more efficiently using addition of arrays, and the fact that np.sin
    accepts and returns numpy arrays:
    >>> def fn(x, y): return np.sin(x + y)  # x and y are numpy arrays
    >>> a, b = np.array([np.pi, 0]), np.array([0, np.pi/2])
    >>> answer = fn(a, b)
    >>> np.allclose(np.array([0, 1]), answer)
    True
"""

from . import proposals
from . import densities
from . import util
from . import phase_space

from .integration import *
from .markov import *

from .sampling import AcceptRejectSampler, Sample
from .density import Proposal, Density, Distribution
