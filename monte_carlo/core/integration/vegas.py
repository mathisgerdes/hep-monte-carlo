import numpy as np
from .integration import IntegrationSample
from .stratified_volume import GridVolumes
from ..util import damped_update


class VegasSample(IntegrationSample):

    def __init__(self, **kwargs):
        self.chi2 = None
        super().__init__(**kwargs)


class VegasMC(object):

    def __init__(self, ndim=1, divisions=1, c=3,
                 name="MC VEGAS", var_weighted=False):
        """ VEGAS Monte Carlo integration algorithm.

        :param ndim: Dimensionality of the integral.
        :param divisions: Number of divisions of the volume along
            each dimension. The total number of 'boxes' is divisions^ndim
        :param c: Damping parameter (greater implies less damping), used in
            adapting the division sizes. See damped_update of this module.
        :param name: Method name used for plotting.
        :param var_weighted: If true, weight the estimates from different
            iterations with their variance (to obtain the best estimate).
            Note that this can lead to a bias if the variances and estimates
            are correlated.
        """
        # the configuration is defined by the sizes of the bins
        self.ndim = ndim
        self.volumes = GridVolumes(ndim=ndim, divisions=divisions)
        # number of bins along each axis
        self.divisions = divisions
        self.c = c
        self.var_weighted = var_weighted
        self.method_name = name

    def get_interface_infer_multiple(self, sub_eval_count):
        """ Construct an interface that only takes fn and a total sample size.

        If the sample size is not an integer multiple of sub_eval_count,
        the actual number of function evaluations might be lower than N.

        :param sub_eval_count: Number of function evaluations per iteration.
        """
        def interface(fn, eval_count, apriori=True, chi=False):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is the VEGAS algorithm.

            :param chi: If true return the chi^2/dof value of the estimates
                obtained in the various iterations.
            :param apriori: If true, reset the division sizes.
            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            # inevitably do fewer evaluations for some eval_count values
            iterations = eval_count // sub_eval_count
            return self(fn, sub_eval_count, iterations, apriori, chi)

        interface.method_name = self.method_name

        return interface

    def weights_at(self, indices):
        vols = np.prod([self.volumes.sizes[d][indices[d]]
                        for d in range(self.ndim)], axis=0)

        return vols * self.volumes.partition_count

    def __call__(self, fn, sub_eval_count, iterations, apriori=True, chi=False):
        """ Approximate the integral of fn using stratified sampling.

        :param fn: Integrand.
        :param sub_eval_count: Number of function evaluations per iteration.
        :param iterations: Number of iterations.
        :param apriori: If true, reset the sizes of the division.
        :param chi: Indicate whether chi^2 over the estimates of all
            iterations should be computed. Can only do this if iterations >= 2.
        :return: If chi=False tuple (integral_estimate, error_estimate),
            otherwise the tuple (integral_estimate, error_estimate, chi^2/dof).
        """
        if apriori:
            # start anew
            self.volumes.reset()

        assert not chi or iterations > 1, "Can only compute chi^2 if there" \
                                          "is more than one iteration "

        sample = IntegrationSample()
        est_j = np.zeros(iterations)  # The estimate in each iteration j
        var_j = np.zeros(iterations)  # sample estimate of the variance of est_j
        # keep track of contributions of each marginalized bin to the estimate
        estimates = np.zeros((self.ndim, self.divisions))

        for j in range(iterations):
            indices = self.volumes.random_bins(sub_eval_count)
            samples_t = self.volumes.sample_indices(indices)
            values = fn(*samples_t)
            weights = self.weights_at(indices)
            weighted = values * weights
            sample.extend_array('data', samples_t)
            sample.extend_array('function_values', values)
            sample.extend_array('weights', weights)
            for div in range(self.divisions):
                estimates[:, div] = np.mean(weighted * (indices == div), axis=1)
            est_j[j] = np.mean(weighted)
            var_j[j] = np.var(weighted) / sub_eval_count

            inv_sizes = damped_update(est_j[j] / self.volumes.sizes,
                                      self.divisions * estimates, self.c, j)
            sizes = 1 / inv_sizes
            # normalize
            sizes /= np.add.reduce(sizes, axis=1)[:, np.newaxis]
            self.volumes.sizes = sizes

        # note: weighting with sub_eval_count here is redundant,
        # but illustrates how this algorithm could be expanded
        # (could modify to make sub_eval_count vary with j)

        if self.var_weighted:
            norm = np.sum(sub_eval_count / var_j)  # normalization factor
            total_est = np.sum(sub_eval_count * est_j / var_j) / norm
            # final estimate: weight by sub_eval_count and var_j
            var = np.sum(sub_eval_count ** 2 / var_j) / norm ** 2
        else:
            total_est = (np.sum(sub_eval_count * est_j) /
                         (iterations * sub_eval_count))
            var = (np.sum(sub_eval_count ** 2 * var_j) /
                   (iterations * sub_eval_count) ** 2)

        sample.integral = total_est
        sample.integral_err = np.sqrt(var)

        if chi:
            # chi^2/dof, have "iteration" values that are combined,
            # so here dof = iterations - 1
            chi2 = np.sum((est_j - total_est) ** 2 / var_j) / (iterations - 1)
            sample.chi2 = chi2

        return sample
