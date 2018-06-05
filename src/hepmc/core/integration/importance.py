import numpy as np
from .integration import IntegrationSample


class ImportanceMC(object):

    def __init__(self, dist, name="MC Importance"):
        """ Importance sampling Monte Carlo integration.

        Importance sampling replaces the uniform sample distribution of plain
        Monte Carlo with a custom pdf.

        By default a uniform probability distribution is used, making the method
        equivalent to plain MC.

        Example:
            >>> from hepmc import densities
            >>> sampling = lambda size: np.random.rand(size)**2
            >>> pdf = lambda x: 2*x
            >>> dist = Density.make(ndim=1, pdf=pdf, rvs=sampling)
            >>> mc_imp = ImportanceMC(1, dist)
            >>> est, err = mc_imp(lambda x: x, 1000)
            >>> est, err  # the pdf is ideal since fn(x)/(2*x) = 1/2 = const
            (0.5, 0.0)

        :param dist: Distribution to use for sampling.
        :param name: Name of the method that can be used as label in
            plotting routines (can be changed to name parameters).
        """
        self.method_name = name

        self.dist = dist
        self.ndim = dist.ndim

    def __call__(self, fn, eval_count):
        """ Approximate the integral of fn.

        :param fn: Integrand, taking ndim numpy arrays and returning a number.
        :param eval_count: Total number of function evaluations.
        :return: Tuple (integral_estimate, error_estimate).
        """
        xs = self.dist.rvs(eval_count)
        ys = fn(*xs.transpose())
        weights = self.dist.pdf(xs)
        sample = IntegrationSample(data=xs, function_values=ys, weights=weights)

        # integral estimate
        sample.integral = np.mean(ys / weights)
        # variance of the weighted function samples
        sample.integral_err = np.sqrt(np.var(ys / weights) / eval_count)
        return sample


class MultiChannelMC(object):

    def __init__(self, channels, b=.5, name="MC Multi C.",
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
            >>> from hepmc import densities
            >>> channels = MultiChannel([densities.Uniform(1)])
            >>> mc_imp = MultiChannelMC(channels)  # same as plain MC
            >>> est, err = mc_imp(lambda x: x, [], [100], [])

        :type channels: MultiChannel
        :param channels: Importance sampling channels used in the integration.
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

    def get_interface_ratios(self, sub_eval_count=100, r1=0, r2=1, r3=0):
        """ Get an interface to the integration that only takes a sample size.

        Fix the ratios of function evaluations spent in each phase and infer
        the multiple of a fixed number sub_eval_count of function evaluations
        for each phase.

        If a given eval_count cannot be split equally into bins of size
        sub_eval_count, the remaining evaluations are spent in phase 3.

        :param sub_eval_count: Number of function evaluations in
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

        def interface(fn, eval_count, apriori=True):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is Multi channel
            Monte Carlo.

            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :param apriori: If true, reset the channel weights for each
                call to the method.
            :return: Tuple (integral_estimate, error_estimate).
            """
            num_iterations = eval_count // sub_eval_count
            m1 = int(r1 * num_iterations)
            m2 = int(r2 * num_iterations)
            m3 = int(r3 * num_iterations)
            samples_remaining = eval_count - (m1 + m2 + m3) * sub_eval_count

            iter_1 = [sub_eval_count] * m1
            iter_2 = [sub_eval_count] * m2
            iter_3 = [sub_eval_count] * m3
            if samples_remaining:
                iter_3.append(samples_remaining)
            return self(fn, iter_1, iter_2, iter_3, apriori=apriori)

        interface.method_name = self.method_name

        return interface

    def iterate(self, fn, eval_count, update_weights=True, get_estimate=True):
        """ One iteration of the algorithm with sample size eval_count.

        :param fn: Integrand.
        :param eval_count: Number of function evaluations in the iteration.
        :param update_weights: If true, channel weights are updated according
            to the respective contributions to the variance.
        :param get_estimate: Specify if integral estimate (i.e. function mean)
            and sample variance should be computed.
        :return: If compute_estimate is true, estimate and contribution to
            sample variance of the estimate w_est. Otherwise return nothing.
            The variance of the estimate is (w_est - est^2) / eval_count
        """
        # a ChannelSample object
        sample = self.channels.sample(eval_count)

        fn_values = fn(*sample.data.transpose())
        weights = sample.weights
        # weighted samples of fn
        weighted = (fn_values / weights)
        # contribution to variance of fn
        w_fn = (np.add.reduceat(weighted ** 2, sample.channel_bounds) /
                sample.count_per_channel)

        if update_weights:
            factors = sample.channel_weights * np.power(w_fn, self.b)
            self.channels.update_channel_weights(factors / np.sum(factors))

        if get_estimate:
            estimate = np.mean(weighted)
            w_est = np.sum(sample.channel_weights * w_fn)
            # return estimate, w_est
            return sample.data, fn_values, weights, estimate, w_est

    def __call__(self, fn, eval_count_1, eval_count_2, eval_count_3,
                 apriori=True):
        """ Approximate the integral of fn over the [0,1]^ndim hypercube.

        :param fn: Integrand.
        :param eval_count_1: List of sample sizes for the iterations
            in phase 1.
        :param eval_count_2: List of sample sizes for the iterations
            in phase 2.
        :param eval_count_3: List of sample sizes for the iterations
            in phase 3.
        :param apriori: If true, reset the channel weights.
        :return: Tuple (integral_estimate, error_estimate).
        """
        if apriori:
            self.channels.reset()

        for eval_count in eval_count_1:
            self.iterate(fn, eval_count, True, False)

        sample = IntegrationSample()
        eval_counts = np.concatenate([eval_count_2, eval_count_3])
        eval_counts = eval_counts.astype(np.int)
        m2 = len(eval_count_2)  # number of iterations in phase 2
        ws_est = np.empty(eval_counts.size)
        estimates = np.empty(eval_counts.size)

        for j, eval_count in zip(range(eval_counts.size), eval_counts):
            xs, values, weights, est, w = self.iterate(fn, eval_count, j < m2)
            estimates[j], ws_est[j] = est, w
            sample.extend_array('function_values', values)
            sample.extend_array('weights', weights)
            sample.extend_array('data', xs)

        if sample.function_values is not None:
            if self.var_weighted:
                # sample variance of individual iterations
                variances = (ws_est - estimates ** 2) / eval_counts
                norm = np.sum(eval_counts / variances)
                total_est = np.sum(eval_counts * estimates / variances) / norm
                var = np.sum(eval_counts ** 2 / variances) / norm ** 2
            else:
                total_evaluations = np.sum(eval_counts)
                total_est = np.sum(estimates * eval_counts) / total_evaluations
                var = (np.sum(eval_counts * ws_est / total_evaluations) -
                       total_est ** 2) / total_evaluations

            sample.integral = total_est
            sample.integral_err = np.sqrt(var)

        return sample
