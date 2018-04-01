""" Module for Monte Carlo integration methods.

The integral for all techniques is performed over the unit hypercube [0, 1]^dim.
The integrands must return one dimensional values (e.g. floats)

The module contains several Monte Carlo integration methods, each of which
is wrapped in a class. To create an integrator for certain method, create
an object specifying general settings such as dimensionality or method-specific
parameters. The object is callable with a function (integrand) and iteration
settings to compute the integral. They return the integral estimate and the
(natural estimator of the) standard deviation of that estimate.

Example:
    >>> mc = MonteCarloPlain()  # defaults to one dimensional integration.
    >>> # Integrate f(x) = x, using 1000 sample points.
    >>> est, err = mc(lambda x: x, 1000)

Advanced variance-reducing Monte Carlo techniques contain several phases,
for each of which the number of iterations and number of function evaluations
need to be specified. Since the structure of the algorithms differ,
the call signature is not the same for all methods.

Example:
    Divide the integration space into 4 equally sized partitions with a base
    number of 10 sample points in each volume.
    >>> volumes = GridVolumes(dim=1, divisions=4, default_count=10)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> # Stratified sampling expects a multiple instead of a total sample size.
    >>> est, err = mc_strat(lambda x: x, 5)  # 5 * 10 sample points per region

To allow sensible comparisons of the different techniques, they all provide
a function for creating an interface where only the total number of function
evaluations (which are considered to be expensive -- the asymptotic efficiency
is usually in this variable) and the integrand are passed.

Example:
    >>> volumes = GridVolumes(dim=1, divisions=4, default_count=100)
    >>> mc_strat = MonteCarloStratified(volumes=volumes)
    >>> mc_strat_ = mc_strat.get_interface_infer_multiple()
    >>> est, err = mc_strat(lambda x: x, 4000)  # multiple is 4000/(4*100) = 40

The integrand passed to a dim-dimensional method must take dim arguments, all
of which are 1d numpy arrays of some size N, and return a numpy array of
the same length N. The numpy vectorize function can help if the integrand of
interest cannot be written using vectors (arrays) easily
(note, however, that vectorize does not increase efficiency).

Example:
    >>> def f(x, y): return np.sin(x + y)  # x and y are floats
    >>> f = np.vectorize(f)                # use @np.vectorize instead
    >>> answer = f([np.pi, 0], [0, np.pi/2])
    >>> np.allclose(np.array([0, 1]), answer)
    True

    Or more efficiently using addition of arrays, and the fact that np.sin
    accepts and returns numpy arrays:
    >>> def f(x, y): return np.sin(x + y)  # x and y are numpy arrays
    >>> a, b = np.array([np.pi, 0]), np.array([0, np.pi/2])
    >>> answer = f(a, b)
    >>> np.allclose(np.array([0, 1]), answer)
    True
"""

import numpy as np
from monte_carlo.sampling import GridVolumes, Channels, assure_2d


# HELPER
def damped_update(old, new, damping_onset, inertia):
    """ Update an old value given a new one, damping the change.

    The parameter inertia can be thought of loosely as the index of the change
    in a number of update iterations, where damping_onset specifies the
    damping-behavior. Both damping_onset and inertia can be floats.

    :param old:
    :param new:
    :param damping_onset: The value of inertia for which old and new values have
        equal weights in the update. Greater values mean smaller damping.
    :param inertia: Value > 0 specifying the strength of damping.
        In the limit to infinity, the old value is retained.
    :return: An updated value combining the old (current) an new values.
    """
    return (new * damping_onset / (inertia + 1 + damping_onset) +  # old
            old * (inertia + 1) / (inertia + 1 + damping_onset))   # new


# MONTE CARLO METHODS
class MonteCarloPlain(object):
    """ Plain Monte Carlo integration method.

    Approximate the integral as the mean of the integrand over a randomly
    selected sample (uniform probability distribution over the unit hypercube).
    """
    def __init__(self, dim=1, name="MC Plain"):
        self.method_name = name
        self.dim = dim

    def __call__(self, f, sample_size):
        """ Compute Monte Carlo estimate of dim-dimensional integral of f.

        The integration volume is the dim-dimensional unit cube [0,1]^dim.

        :param f: A function accepting self.dim numpy arrays, returning an array
            of the same length with the function values.
        :param sample_size: Total number of function evaluations used to
            approximate the integral.
        :return: Tuple (integral_estimate, error_estimate) where
            the error_estimate is based on the unbiased sample variance
            of the function, computed on the same sample as the integral.
            According to the central limit theorem, error_estimate approximates
            the standard deviation of the statistical (normal) distribution
            of the integral estimates.
        """
        x = np.random.rand(self.dim * sample_size).reshape(sample_size,
                                                           self.dim)
        y = f(*x.transpose())
        mean = np.mean(y)
        var = np.var(y)
        return mean, np.sqrt(var / sample_size)


class MonteCarloImportance(object):

    def __init__(self, dim=1, sampling=None, pdf=lambda *x: 1,
                 name="MC Importance"):
        """ Importance sampling Monte Carlo integration.

        Importance sampling replaces the uniform sample distribution of plain
        Monte Carlo with a custom pdf.

        By default a uniform probability distribution is used, making the method
        equivalent to plain MC.

        Example:
            >>> sampling = lambda size: np.random.rand(size)**2
            >>> pdf = lambda x: 2*x
            >>> mc_imp = MonteCarloImportance(1, sampling, pdf)
            >>> est, err = mc_imp(lambda x: x, 1000)
            >>> est, err  # the pdf is ideal since f(x)/(2*x) = 1/2 = const
            (0.5, 0.0)

        :param dim: Dimensionality of the integral.
        :param sampling: Function taking a number N as argument and returning
            that number of samples distributed according to the pdf.
            It must return a numpy array of shape (N, dim).
            If dim=1, sampling may also return an array of shape (N,) instead.
        :param pdf: Distributions of samples used to approximate the integral.
            The function must take dim numpy arrays of arbitrary but equal
            lengths. The i-th array specifies the coordinates of the
            i-th dimension for all sample points.
        :param name: Name of the method that can be used as label in
            plotting routines (can be changed to name parameters).
        """
        self.method_name = name
        self.dim = dim

        if sampling is None:
            # default is a uniform distribution
            def sampling(sample_size):
                """ Uniform sampling of sample_size dim-dimensional values. """
                sample = np.random.rand(sample_size * self.dim)
                return sample.reshape(sample_size, self.dim)

        self.sampling = sampling
        self.pdf = pdf

    def __call__(self, f, sample_size):
        """ Approximate the integral of f.

        :param f: Integrand, taking dim numpy arrays and returning a number.
        :param sample_size: Total number of function evaluations.
        :return: Tuple (integral_estimate, error_estimate).
        """
        x = assure_2d(self.sampling(sample_size))

        y = f(*x.transpose()) / self.pdf(*x.transpose())

        mean = np.mean(y)  # integral estimate
        var = np.var(y)    # variance of the weighted function samples
        return mean, np.sqrt(var / sample_size)


class MonteCarloMultiImportance(object):

    def __init__(self, channels: Channels, b=.5, name="MC Multi C.",
                 var_weighted=False):
        """ Multi channel Monte Carlo integration.

        Use multiple importance sampling channels to approximate the integral.
        Each of the channels has a weight that is adapted and optimized
        throughout the integration.

        The integration is split into three phases, each of which containing
        several iterations (of which each contains several function
        evaluations).

        Phase 1: Update weights but discard integral estimates.
        Phase 2: Update weights and use the results for the integral estimation.
        Phase 3: Don't update weights and continue using the current weights to
            estimate the integral.

        Example:
            >>> uniform_pdf = lambda *x: 1
            >>> uniform_sampling = lambda size: np.random.rand(size)
            >>> channels = Channels([uniform_sampling], [uniform_pdf])
            >>> mc_imp = MonteCarloMultiImportance(channels)  # same as plain MC
            >>> est, err = mc_imp(lambda x: x, [], [100], [])

        :param channels: (Importance sampling) Channels used in the integration.
        :param b: Exponent between 1/2 and 1/4, used in updating the weights.
        :param name: Name of the method used for plotting.
        :param var_weighted: If true, weight the estimates from different
            iterations with their variance (to obtain the best estimate).
            Note that this can lead to a bias if the variances and estimates
            are correlated.
        """
        self.method_name = name
        self.channels = channels
        self.var_weighted = var_weighted
        self.b = b

    def get_interface_ratios(self, sub_sample_size=100, r1=0, r2=1, r3=0):
        """ Get an interface to the integration that only takes a sample size.

        Fix the ratios of function evaluations spent in each phase and infer
        the multiple of a fixed number sub_sample_size of function evaluations
        for each phase.

        If a given sample_size cannot be split equally into bins of size
        sub_sample_size, the remaining evaluations are spent in phase 3.

        :param sub_sample_size: Number of function evaluations in
            each iteration.
        :param r1: Gives the ratio of iterations spent in the first phase.
        :param r2: Ratio of second phase (by default 1-r1-r3).
        :param r3: Ratio of third phase (by default 0).
        :return: A function serving as interface.
        """
        if r1 < 0 or r2 < 0 or r3 < 0:
            raise ValueError(
                "Ratios cannot be smaller than 0: %d, %d, %d." % (r1, r2, r3))
        if not np.isclose(1, r1 + r2 + r3):
            raise ValueError("Ratios must sum to 1: %d + %d + %d = %d." % (
                r1, r2, r3, r1 + r2 + r3))

        def interface(f, sample_size, apriori=True):
            """ Approximate the integral of f via using given sample size.

            The method used to approximate the integral is Multi channel
            Monte Carlo.

            :param f: Integrand.
            :param sample_size: Total number of function evaluations.
            :param apriori: If true, reset the channel weights for each
                call to the method.
            :return: Tuple (integral_estimate, error_estimate).
            """
            num_iterations = sample_size // sub_sample_size
            m1 = int(r1 * num_iterations)
            m2 = int(r2 * num_iterations)
            m3 = int(r3 * num_iterations)
            samples_remaining = sample_size - (m1 + m2 + m3) * sub_sample_size

            iter_1 = [sub_sample_size] * m1
            iter_2 = [sub_sample_size] * m2
            iter_3 = [sub_sample_size] * m3
            if samples_remaining:
                iter_3.append(samples_remaining)
            return self(f, iter_1, iter_2, iter_3, apriori=apriori)

        interface.method_name = self.method_name

        return interface

    def iterate(self, f, sample_size, update_weights=True, get_estimate=True):
        """ One iteration of the algorithm with sample size sample_size.

        :param f: Integrand.
        :param sample_size: Number of function evaluations in the iteration.
        :param update_weights: If true, channel weights are updated according
            to the respective contributions to the variance.
        :param get_estimate: Specify if integral estimate (i.e. function mean)
            and sample variance should be computed.
        :return: If compute_estimate is true, estimate and contribution to
            sample variance of the estimate w_est. Otherwise return nothing.
            The variance of the estimate is (w_est - est^2) / sample_size
        """
        # a ChannelSample object
        sample = self.channels.generate_sample(sample_size)

        # weighted samples of f
        f_samples = (f(*sample.sample.transpose()) / sample.sample_weights)
        # contribution to variance of f
        w_f = (np.add.reduceat(f_samples ** 2, sample.channel_bounds) /
               sample.count_per_channel)

        if update_weights:
            factors = sample.channel_weights * np.power(w_f, self.b)
            self.channels.update_channel_weights(factors / np.sum(factors))

        if get_estimate:
            estimate = np.mean(f_samples)
            w_est = np.sum(sample.channel_weights * w_f)
            return estimate, w_est

    def __call__(self, f, sample_sizes_1, sample_sizes_2, sample_sizes_3,
                 apriori=True):
        """ Approximate the integral of f over the [0,1]^dim hypercube.

        :param f: Integrand.
        :param sample_sizes_1: List of sample sizes for the iterations
            in phase 1.
        :param sample_sizes_2: List of sample sizes for the iterations
            in phase 2.
        :param sample_sizes_3: List of sample sizes for the iterations
            in phase 3.
        :param apriori: If true, reset the channel weights.
        :return: Tuple (integral_estimate, error_estimate).
        """
        if apriori:
            self.channels.reset()

        for sample_size in sample_sizes_1:
            self.iterate(f, sample_size, True, False)

        sample_sizes = np.concatenate([sample_sizes_2, sample_sizes_3])
        sample_sizes = sample_sizes.astype(np.int)
        m2 = len(sample_sizes_2)  # number of iterations in phase 2
        ws_est = np.empty(sample_sizes.size)
        estimates = np.empty(sample_sizes.size)

        for j, sample_size in zip(range(sample_sizes.size), sample_sizes):
            estimates[j], ws_est[j] = self.iterate(f, sample_size, j < m2, True)

        if self.var_weighted:
            # sample variance of individual iterations
            variances = (ws_est - estimates ** 2) / sample_sizes
            norm = np.sum(sample_sizes / variances)
            total_est = np.sum(sample_sizes * estimates / variances) / norm
            var = np.sum(sample_sizes ** 2 / variances) / norm ** 2
        else:
            total_sample_size = np.sum(sample_sizes)
            total_est = np.sum(estimates * sample_sizes) / total_sample_size
            var = (np.sum(sample_sizes * ws_est / total_sample_size) -
                   total_est ** 2) / total_sample_size

        return total_est, np.sqrt(var)


# STRATIFIED
class MonteCarloStratified(object):

    def __init__(self, volumes=GridVolumes(), name="MC Stratified"):
        """ Stratified Monte Carlo integration.

        Approximate N-dimensional integral of f over the unit hypercube
        by sampling the function independently in a specified division of
        the integration volume.

        :param volumes: Partition of the unit hypercube volume.
            If using a Volumes class other than GridVolumes, it must implement
            a method iterate that behaves as the one in GridVolumes.
        :param name: Name of the method used for plotting.
        """
        self.method_name = name
        self.dim = volumes.dim
        self.volumes = volumes

    def get_interface_infer_multiple(self):
        """ Construct an interface that only takes f and a total sample size.

        If the sample size is not an integer multiple of
        self.volumes.total_base_size, the actual number of function evaluations
        might be lower than N.
        """
        def interface(f, sample_size):
            """ Approximate the integral of f via using given sample size.

            The method used to approximate the integral is stratified MC.

            :param f: Integrand.
            :param sample_size: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            return self(f, sample_size / self.volumes.total_base_size)

        interface.method_name = self.method_name

        return interface

    def __call__(self, f, multiple):
        """ Approximate the integral of f using stratified sampling MC.

        :param f: Integrand.
        :param multiple: Multiply the base sample size of each region with this
            number. The total number of function evaluations will then be
            multiple * self.volumes.total_base_size.
        :return: Tuple (integral_estimate, error_estimate).
        """
        int_est = 0
        var_est = 0
        for sub_sample_size, sample, vol in self.volumes.iterate(multiple):
            values = f(*sample.transpose())
            int_est += vol * np.mean(values)
            var_est += np.var(values) * vol ** 2 / sub_sample_size
        return int_est, np.sqrt(var_est)


# VEGAS
class MonteCarloVEGAS(object):

    def __init__(self, dim=1, divisions=1, c=3,
                 name="MC VEGAS", var_weighted=False):
        """ VEGAS Monte Carlo integration algorithm.


        :param dim: Dimensionality of the integral.
        :param divisions: Number of divisions of the volume along
            each dimension. The total number of 'boxes' is divisions^dim
        :param c: Damping parameter (greater implies less damping), used in
            adapting the division sizes. See damped_update of this module.
        :param name: Method name used for plotting.
        :param var_weighted: If true, weight the estimates from different
            iterations with their variance (to obtain the best estimate).
            Note that this can lead to a bias if the variances and estimates
            are correlated.
        """
        # the configuration is defined by the sizes of the bins
        self.sizes = np.ones((dim, divisions)) / divisions
        self.dim = dim
        self.volumes = GridVolumes(dim=dim, divisions=divisions)
        # number of bins along each axis
        self.divisions = divisions
        self.c = c
        self.var_weighted = var_weighted
        self.method_name = name

    def get_interface_infer_multiple(self, sub_sample_size):
        """ Construct an interface that only takes f and a total sample size.

        If the sample size is not an integer multiple of sub_sample_size,
        the actual number of function evaluations might be lower than N.

        :param sub_sample_size: Number of function evaluations per iteration.
        """
        def interface(f, sample_size, apriori=True, chi=False):
            """ Approximate the integral of f via using given sample size.

            The method used to approximate the integral is the VEGAS algorithm.

            :param chi: If true return the chi^2/dof value of the estimates
                obtained in the various iterations.
            :param apriori: If true, reset the division sizes.
            :param f: Integrand.
            :param sample_size: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            # inevitably do fewer evaluations for some sample_size values
            iterations = sample_size // sub_sample_size
            return self(f, sub_sample_size, iterations, apriori, chi)

        interface.method_name = self.method_name

        return interface

    def choice(self):
        """ Return the multi-index of a random bin. """
        return np.random.randint(0, self.divisions, self.dim)

    def pdf(self, indices):
        """ Probability density of finding a point in bin with given index. """
        # since here N = 1 = const for all bins,
        # the pdf is given simply by the volume of the regions
        vol = np.prod([self.sizes[d][indices[d]] for d in range(self.dim)],
                      axis=0)
        return 1 / self.divisions ** self.dim / vol

    def plot_pdf(self):
        """ Plot the pdf resulting from the current bin sizes. """
        self.volumes.plot_pdf(label="VEGAS pdf")

    def __call__(self, f, sub_sample_size, iterations, apriori=True, chi=False):
        """ Approximate the integral of f using stratified sampling.

        :param f: Integrand.
        :param sub_sample_size: Number of function evaluations per iteration.
        :param iterations: Number of iterations.
        :param apriori: If true, reset the sizes of the division.
        :param chi: Indicate whether chi^2 over the estimates of all
            iterations should be computed. Can only do this if iterations >= 2.
        :return: If chi=False tuple (integral_estimate, error_estimate),
            otherwise the tuple (integral_estimate, error_estimate, chi^2/dof).
        """
        if apriori:
            # start anew
            self.sizes = np.ones((self.dim, self.divisions)) / self.divisions
            self.volumes.update_bounds_from_sizes(self.sizes)

        assert not chi or iterations > 1, "Can only compute chi^2 if there" \
                                          "is more than one iteration "

        est_j = np.zeros(iterations)  # The estimate in each iteration j
        var_j = np.zeros(iterations)  # sample estimate of the variance of est_j
        # keep track of contributions of each marginalized bin to the estimate
        estimates = np.zeros((self.dim, self.divisions))

        for j in range(iterations):
            indices = self.volumes.random_bins(sub_sample_size)
            samples_t = self.volumes.sample(indices)
            values = f(*samples_t) / self.pdf(indices)
            for div in range(self.divisions):
                estimates[:, div] = np.mean(values * (indices == div), axis=1)
            est_j[j] = np.mean(values)
            var_j[j] = np.var(values) / sub_sample_size

            inv_sizes = damped_update(est_j[j] / self.sizes,
                                      self.divisions * estimates, self.c, j)
            self.sizes = 1 / inv_sizes
            # normalize
            self.sizes /= np.add.reduce(self.sizes, axis=1)[:, np.newaxis]
            self.volumes.update_bounds_from_sizes(self.sizes)

        # note: weighting with sub_sample_size here is redundant,
        # but illustrates how this algorithm could be expanded
        # (could modify to make sub_sample_size vary with j)

        if self.var_weighted:
            norm = np.sum(sub_sample_size / var_j)  # normalization factor
            total_est = np.sum(sub_sample_size * est_j / var_j) / norm
            # final estimate: weight by sub_sample_size and var_j
            var = np.sum(sub_sample_size ** 2 / var_j) / norm ** 2
        else:
            total_est = (np.sum(sub_sample_size * est_j) /
                         (iterations * sub_sample_size))
            var = (np.sum(sub_sample_size ** 2 * var_j) /
                   (iterations * sub_sample_size) ** 2)

        if chi:
            # chi^2/dof, have "iteration" values that are combined,
            # so here dof = iterations - 1
            chi2 = np.sum((est_j - total_est) ** 2 / var_j) / (iterations - 1)
            return total_est, np.sqrt(var), chi2

        return total_est, np.sqrt(var)
