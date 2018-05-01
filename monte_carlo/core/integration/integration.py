import numpy as np

from ..sampling import Sample


class IntegrationSample(Sample):

    def __init__(self, **kwargs):
        self.function_values = None

        # computed by the integration methods
        self.integral = None
        self.integral_err = None

        super().__init__(**kwargs)


class PlainMC(object):
    """ Plain Monte Carlo integration method.

    Approximate the integral as the mean of the integrand over a randomly
    selected sample (uniform probability distribution over the unit hypercube).
    """
    def __init__(self, ndim=1, name="MC Plain"):
        self.method_name = name
        self.ndim = ndim

    def __call__(self, fn, eval_count):
        """ Compute Monte Carlo estimate of ndim-dimensional integral of fn.

        The integration volume is the ndim-dimensional unit cube [0,1]^ndim.

        :param fn: A function accepting self.ndim numpy arrays,
            returning an array of the same length with the function values.
        :param eval_count: Total number of function evaluations used to
            approximate the integral.
        :return: Tuple (integral_estimate, error_estimate) where
            the error_estimate is based on the unbiased sample variance
            of the function, computed on the same sample as the integral.
            According to the central limit theorem, error_estimate approximates
            the standard deviation of the statistical (normal) distribution
            of the integral estimates.
        """
        sample = IntegrationSample()
        sample.data = np.random.random((eval_count, self.ndim))
        sample.function_values = fn(*sample.data.transpose())
        sample.integral = np.mean(sample.function_values)
        err = np.sqrt(np.var(sample.function_values) / eval_count)
        sample.integral_err = err
        return sample
