import numpy as np
import matplotlib.pyplot as plt
from monte_carlo.sampling import *


# HELPER
def damped_update(old, new, c, j):
    return new * (c)/(j+1+c) + (j+1)/(j+1+c) * old


# MONTE CARLO METHODS
class MonteCarloPlain(object):

    def __init__(self, dim=1, name="MC Plain"):
        self.method_name = name
        self.dim = dim

    def __call__(self, f, N):
        """ Compute Monte Carlo estimate of N-dimensional integral of f over the unit hypercube.

        Args:
            f: A function accepting self.dim numpy arrays, returning an array of the same length
                with the function values. For function that cannot simply be factorized,
                consider to use @numpy.vectorize before the function definition.
        Returns:
            tuple (integral_estimate, error_estimate),
            where the error_estimate is based on the unbiased sample variance of the function,
            computed on the same sample as the integral.
        """
        # using this order of N, dim to be consistent with other algorithms
        # also less awkward if N=1, compare [[1, 2]] vs [[1], [2]]
        x = np.random.rand(self.dim*N).reshape(N, self.dim)
        y = f(*x.transpose())
        mean = np.mean(y)
        var = np.var(y)
        return mean, np.sqrt(var/N)


class MonteCarloImportance(object):
    """
    By default, uniform probability distribution is used. pdf is weight function given the sampling distribution.

    Args:
        sampling_fn: Function taking one argument N, random number according to the desired distribution,
            i.e. an array of shape (N, dim).
            If dim = 1, may also return a simple numpy array of length N.
        pdf: Function with the same signature as the integrand. Takes dim numpy arrays of size N and returns
            one numpy array of size N giving the probability density of the sampling funciton for each point.

    """
    def __init__(self, dim=1, sampling_fn=None, pdf=lambda *x:1, name="MC Importance"):
        self.method_name = name

        self.dim = dim
        if sampling_fn is None:
            sampling_fn = lambda N: np.random.rand(N * self.dim).reshape(N, self.dim)
        self.sampling_fn = sampling_fn
        self.pdf = pdf

    def __call__(self, f, N):
        x = self.sampling_fn(N)
        # allow x to be a simple array if dim=1
        if x.ndim == 1:
            x = x[:, np.newaxis]

        y = f(*x.transpose()) / self.pdf(*x.transpose())

        mean = np.mean(y)  # integral estimate
        var = np.var(y)    # variance of the weighted function samples
        return mean, np.sqrt(var/N)


class MonteCarloMultiImportance(object):
    def __init__(self, channels, Nj=100, m1=0, m2=1, m3=0, b=.5, name="MC Imp. Mult.", var_weighted=False):
        """

        The total number of integrand evaluations is N = (m1 + m2 + m3) * Nj

        The integration is split into three phases, each of which containing several iterations.

        Phase 1: Update weights but discrad integral estimates.
        Phase 2: Update weights and use the results for the integral estimation.
        Phase 3: Don't update weights and continue using the current weights to estimate the integral.

        Args:
            b: exponent between 1/2 and 1/4, used in updating the weights.
            var_weighted: Enabling weighting the estimate with the individual variances.
        """
        self.method_name = name
        self.channels = channels
        self.var_weighted = var_weighted
        self.b = b

    def get_interface_const_iterations(self, Nj=100, m1=0, m2=1, m3=0):
        """ Get a function that estimates the integral given the function and a total number of evaluations.

        Args:
            Nj: Number of function evaluations in each iteration in each phase
            m1: Iterations in phase 1.
            m2: Iterations in phase 2.
            m3: Iterations in phase 3.
        """
        def interface(f, N, apriori=True):
            return self(f, [Nj]*m1, [Nj]*m2, [Nj]*m3, apriori=apriori)
        interface.method_name = self.method_name

        return interface

    def get_interface_ratios(self, Nj=100, r1=0, r2=1, r3=0, b=.5):
        """ Get a function that estimates the integral given the function and a total number of evaluations.

        If a given N cannot be split equally into bins of size Nj,
        spend the remaining iterations in phase 3.

        Args:
            r1: gives the ratio of iterations spent in the first phase
            r2: ratio of second phase (by default 1-r1-r3)
            r3: ratio of third phase (by default 0)
        """
        if r1 < 0 or r2 < 0 or r3 < 0:
            raise ValueError("Ratios cannot be smaller than 0: %d, %d, %d."%(r1, r2, r3))
        if not np.isclose(1, r1+r2+r3):
            raise ValueError("Ratios must sum to 1: %d + %d + %d = %d."%(r1, r2, r3, r1+r2+r3))

        def interface(f, N, apriori=True):
            num_iterations = N // Nj
            m1 = int(r1 * num_iterations)
            m2 = int(r2 * num_iterations)
            m3 = int(r3 * num_iterations)
            N_remaining = N - (m1 + m2 + m3) * Nj

            N3 = [Nj]*m3
            if N_remaining:
                N3.append(N_remaining)
            return self(f, [Nj]*m1, [Nj]*m2, N3, apriori=apriori)
        interface.method_name = self.method_name

        return interface

    def iterate(self, f, N, update_weights=True, compute_estimate=True):
        """
        One iteration of the algorithm with sample size N.

        Args:
            update_weights: If true, channel weights are updated according to the sample.
            compute_estimate: Specify if integral estimate (i.e. function mean) and sample variance
                should be computed.
        Returns:
            If compute_estimate is true, estimate and sample variance of estimate.
        """
        self.channels.generate_sample(N)

        # weighted samples of f
        f_samples = f(*self.channels.samples.transpose()) / self.channels.sample_weights
        Wi = np.add.reduceat(f_samples**2, self.channels.sample_bounds)  / self.channels.sample_sizes

        if update_weights:
            factors = self.channels.sample_cweights * np.power(Wi, self.b)
            self.channels.update_sample_cweights(factors / np.sum(factors))

        if compute_estimate:
            E = np.mean(f_samples)
            W = np.sum(self.channels.sample_cweights * Wi)
            return E, W

    def __call__(self, f, N_phase1, N_phase2, N_phase3, apriori=True):
        """ Approximate the integral of f over the [0,1]^dim hypercube.

        Args:
            N_phase1: List giving sample sizes for each iteration in phase 1.
            N_phase1: List giving sample sizes for each iteration in phase 2.
            N_phase1: List giving sample sizes for each iteration in phase 3.
        """
        if apriori:
            self.channels.reset()

        for N in N_phase1:
            self.iterate(f, N, update_weights=True, compute_estimate=False)

        Ns = np.concatenate([N_phase2, N_phase3]).astype(np.int)
        m2 = len(N_phase2)
        Ws = np.empty(Ns.size)
        Es = np.empty(Ns.size)

        for j, N in zip(range(Ns.size), Ns):
            Es[j], Ws[j] = self.iterate(f, N, update_weights=j<m2, compute_estimate=True)

        if self.var_weighted:
            variances = (Ws - Es**2) / N  # sample variance of individual iterations
            norm = np.sum(Ns / variances)
            E = np.sum(Ns * Es / variances) / norm
            var = np.sum(Ns**2 / variances) / norm**2
        else:
            norm = np.sum(np.sqrt(Ns))
            N = np.sum(Ns)
            E = np.sum(Es * np.sqrt(Ns)) / norm
            var = (np.sum(Ns * Ws / N) - E**2) / N

        return E, np.sqrt(var)


# STRATIFIED
class MonteCarloStratified(object):
    """ Compute Monte Carlo estimate of N-dimensional integral of f over the unit hypercube using strafield sampling.

    Note: N must be an integer multiple of volumes.totalN !

    Returns:
        tuple (integral_estimate, error_estimate)
    """
    def __init__(self, volumes=GridVolumes(), name="MC Stratified"):
        self.method_name = name
        self.dim = volumes.dim
        self.volumes = volumes

    def get_interface_infer_multiple(self):
        """
        Note: If N is not an integer multiple of self.volumes.totalN, the actual
            number of function evaluations might be lower than N.
        """
        def interface(f, N):
            return self(f, N / self.volumes.totalN)
        interface.method_name = self.method_name

        return interface

    def __call__(self, f, multiple):
        int_est = 0
        var_est = 0
        for Nj, sample, vol in self.volumes.iterate(multiple):
            values = f(*sample.transpose())
            int_est += vol * np.mean(values)
            var_est += np.var(values) * vol**2 / Nj
        return int_est, np.sqrt(var_est)

# VEGAS
class MonteCarloVEGAS(object):
    def __init__(self, Nj=100, dim=1, divisions=1, c=3, name="MC VEGAS"):
        """
        c: each iteration, the bin sizes are a combination of old and new bin sizes.
            the weight of the old bin sizes increases with the number of iterations.
            c gives the iteration where both have equal weight (=1 -> equal in first iteration).
            therefore: larger c means the bins change more, smaller means the bins tend to stay close to uniform.
        """
        self.dim = dim
        self.sizes = np.ones((dim, divisions))/divisions  # the configuration is defined by the sizes of the bins
        self.Nj = Nj  # number of samples per iteration
        self.divisions = divisions  # number of bins along each axis
        self.c = c  # measure of damping (smaller means more damping)
        self.method_name = name

    def choice(self):
        """ Return a random choice of bin, specified by its multi-index. """
        return np.random.randint(0, self.divisions, self.dim)

    def random_x(self, bin_index):
        """ Return a random number in the bin of given multi-index. """
        x = np.empty(self.dim)
        for i in range(self.dim):
            x[i] = np.sum(self.sizes[i][:bin_index[i]])
        x += self.sizes[np.arange(self.dim), bin_index] * np.random.rand(self.dim)
        return x

    def pdf(self, bin_index):
        """ Give the probability density of finding a point in bin with given index. """
        # 1 / volume
        return 1/np.power(self.divisions, self.dim) / np.prod(self.sizes[np.arange(self.dim), bin_index])

    def plot_pdf(self):
        """ Plot the pdf resulting from the current bin sizes. """
        assert self.dim == 1, "Can only plot 1D pdf"
        xs = []
        ys = []
        x = 0
        for i, size in zip(range(self.divisions), self.sizes[0]):
            xs.append(x)
            y = 1/self.divisions/size
            ys.append(y)
            x += size
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, label="VEGAS pdf")

    def __call__(self, f, N, apriori=True, xhi=False):
        if apriori:
            # start anew
            self.sizes = np.ones((self.dim, self.divisions))/self.divisions

        assert N >= self.Nj, "N must be at least Nj (default 100) initially set."

        Nj = self.Nj
        m = N // Nj       # number of iterations (the true N is smaller or equal the passed N parameter!)
        Ej = np.zeros(m)  # The estimate in each iteration j
        Sj = np.zeros(m)  # sample variance in each iteration j, estimating the variance of Ej

        for j in range(m):
            Ei = np.zeros((self.dim, self.divisions))
            Ni = np.zeros((self.dim, self.divisions))  # keep track of number of samples in each "projected" bin
            for i in range(Nj):
                bin_index = self.choice()
                x = self.random_x(bin_index)
                f_sample = f(x) / self.pdf(bin_index)
                Ej[j] +=  f_sample / Nj
                Sj[j] += (f_sample)**2 / Nj
                # in each dimension add the contribution to the given bin index (so in total add in self.dim places)
                Ei[np.arange(self.dim), bin_index] += f_sample / Nj

            # new size = 1/(old value * (getting larger with j) + best guess new * (getting smaller with j))
            self.sizes = 1/damped_update(Ej[j]/self.sizes, self.divisions * Ei, self.c, j)
            self.sizes = self.sizes / np.add.reduce(self.sizes, axis=1)[:, np.newaxis] # normalize

        Sj -= Ej**2             # at this point Sj is the sample variance of f/pdf
        Sj = Sj / Nj            # this completes the computation of Sj (estimated variance of Ej)

        C = np.sum(Nj/Sj)       # normalization factor
        E = np.sum(Nj*Ej/Sj)/C  # final estimate of e: weight by Nj and Sj (note: could modify to make Nj vary with j)
        if xhi:
            # xhi^2/dof, have m values that are combined so here dof = m - 1
            xhi2 = np.sum((Ej - E)**2/Sj)/(m-1)
            return E, np.sqrt(np.sum(Nj**2/Sj)/C**2), xhi2
        else:
            return E, np.sqrt(np.sum(Nj**2/Sj)/C**2)


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class MC3(object):
    def __init__(self, dim, channels, fn, delta=None, initial_value=np.random.rand()):
        self.channels = channels
        self.mc_importance = MonteCarloMultiImportance(channels)
        self.fn = fn
        self.dim = dim

        self.sample_IS = MetropolisHasting(initial_value, self.fn, dim,
                                           lambda s, c: self.channels.pdf(c),
                                           lambda s: self.channels.sample(1)[0])
        self.sample_METROPOLIS = Metropolis(initial_value, self.fn, dim,
                                            self.generate_local)

        if np.ndim(delta) == 0:
            delta = np.ones(dim) * delta
        elif delta is None:
            delta = np.ones(dim) * .05
        elif len(delta) == dim:
            delta = np.array(delta)
        else:
            raise ValueError("delta must be None, a float, or an array of length dim.")
        self.delta = delta

        self.accept_min = 0.25
        self.accept_max = 0.5
        self.accept_mean = (0.25 + 0.5)/2

    def generate_local(self, state):
        zero = np.zeros(self.dim)
        one = np.ones(self.dim)
        return np.minimum(np.maximum(zero, state-self.delta/2), one-self.delta) + np.random.rand()*self.delta

    def __call__(self, Ns_integration, N_sample, beta, batch_size=None):
        if batch_size is None:
            batch_size = int((1 - beta) * N_sample / 10)

        self.integral, self.integral_var = self.mc_importance(fn, *Ns_integration)

        sample = np.empty((N_sample, self.dim))
        for i in range(N_sample):
            if np.random.rand() <= beta:
                self.sample_METROPOLIS.state = sample[i] = self.sample_IS(1)
            else:
                self.sample_IS.state = sample[i] = self.sample_METROPOLIS(1)

        return sample
