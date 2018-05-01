import numpy as np
from .integration import IntegrationSample
from .stratified_volume import GridVolumes


class StratifiedSample(IntegrationSample):
    def __init__(self, **kwargs):
        self.sub_sizes = []
        self.sub_weights = []
        super().__init__(**kwargs)

    @property
    def weights(self):
        if self.function_values is None or self.sub_weights == []:
            return None
        weights = np.empty_like(self.function_values)
        index = 0
        for size, weight in zip(self.sub_sizes, self.sub_weights):
            weights[index:index+size] = weight
        return weights

    @weights.setter
    def weights(self, value):
        pass


class StratifiedMC(object):

    def __init__(self, volumes=GridVolumes(), name="MC Stratified"):
        """ Stratified Monte Carlo integration.

        Approximate N-dimensional integral of fn over the unit hypercube
        by sampling the function independently in a specified division of
        the integration volume.

        :param volumes: Partition of the unit hypercube volume.
            If using a Volumes class other than GridVolumes, it must implement
            a method iterate that behaves as the one in GridVolumes.
        :param name: Name of the method used for plotting.
        """
        self.method_name = name
        self.ndim = volumes.ndim
        self.volumes = volumes

    def get_interface_infer_multiple(self):
        """ Construct an interface that only takes fn and a total sample size.

        If the sample size is not an integer multiple of
        self.volumes.total_base_count, the actual number of function evaluations
        might be lower than N.
        """
        def interface(fn, eval_count):
            """ Approximate the integral of fn via using given sample size.

            The method used to approximate the integral is stratified MC.

            :param fn: Integrand.
            :param eval_count: Total number of function evaluations.
            :return: Tuple (integral_estimate, error_estimate).
            """
            return self(fn, eval_count / self.volumes.total_base_count)

        interface.method_name = self.method_name

        return interface

    def __call__(self, fn, multiple):
        """ Approximate the integral of fn using stratified sampling MC.

        :param fn: Integrand.
        :param multiple: Multiply the base sample size of each region with this
            number. The total number of function evaluations will then be
            multiple * self.volumes.total_base_count.
        :return: Tuple (integral_estimate, error_estimate).
        """
        sample = StratifiedSample()
        total_count = self.volumes.total_base_count * multiple
        int_est = 0
        var_est = 0
        for sub_eval_count, xs, vol in self.volumes.iterate(multiple):
            values = fn(*xs.transpose())
            int_est += vol * np.mean(values)
            var_est += np.var(values) * vol ** 2 / sub_eval_count
            sample.sub_sizes.append(sub_eval_count)
            sample.sub_weights.append(sub_eval_count / vol / total_count)
            sample.extend_array('data', xs)
            sample.extend_array('function_values', values)

        sample.integral = int_est
        sample.integral_err = np.sqrt(var_est)
        return sample
