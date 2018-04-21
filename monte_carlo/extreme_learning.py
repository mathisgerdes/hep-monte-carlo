import numpy as np
from monte_carlo.sampling import assure_2d


class FunctionBasis(object):

    def __init__(self, dim):
        self.dim = dim

    def output_matrix(self, xs, params):
        raise NotImplementedError

    def random_params(self, node_count):
        raise NotImplementedError

    def eval(self, xs, params, out_weights):
        return np.dot(self.output_matrix(xs, params), out_weights)

    def extreme_learning_train(self, xs, values, node_count, params=None):
        xs = assure_2d(xs)
        if params is None:
            params = self.random_params(node_count)

        out_matrix = self.output_matrix(xs, params)
        pinv = np.linalg.pinv(out_matrix)

        weights = np.dot(pinv, values)
        return params, weights.flatten()


class AdditiveBasis(FunctionBasis):

    def __init__(self, dim, weight_range=(0, 1), bias_range=(0, -1)):
        """ Linear combination of input followed by non-linear function.

        This class uses the model for additive activation functions from
        "Hamiltonian Monte Carlo acceleration using surrogate
        functions with random bases" (ArXiv ID: 1506.05555):

        If the input is xi and the input weights are wi with biases bi
        (the index i corresponds to the i-th node), the output is
        z(xi) = sum_i gi(wi * xi + bi),
        where the wi and xi are all dim-dimensional and bi are scalars,
        gi are the activation functions.


        Note that the behaviour is different from Radial Base functions, since
        the position of the "centers" do not only depend on the biases.
        The weights do scale the base function, however it is only applied
        linearly in one dimension.

        Assuming all gi = g are the same and g has a center at 0 (as they would
        be for radial base functions). Then the center for node i would be at
        bi / wi. This behavior can be problematic if g is radial/centered,
        as the biases can not be chosen just based on the x-space region.
        Additionally the bounds for the input biases could not be chosen just
        based on the x-space, thus additive base functions are not a substitute
        for radial base functions.

        :param dim: Dimensionality of variable space (value space is 1D)
        :param weight_range: Range the weight can have.
        :param bias_range: Range the input bias can have.
        """
        super().__init__(dim)

        self.include_zero_weight = False

        self.weight_min = weight_range[0]
        self.weight_delta = weight_range[1] - weight_range[0]

        self.bias_min = bias_range[0]
        self.bias_delta = bias_range[1] - bias_range[0]

    def get_outputs(self, inputs, fn_params):
        raise NotImplementedError

    def random_fn_params(self, node_count):
        raise NotImplementedError

    def output_matrix(self, xs, params):
        xs = assure_2d(xs)
        biases, in_weights, fn_params = params

        # inputs: node_count * dim
        inputs = biases[np.newaxis, :] + np.dot(xs, in_weights.transpose())

        outputs = self.get_outputs(inputs, fn_params)
        return outputs

    def random_params(self, node_count):
        biases = self.bias_min + self.bias_delta * np.random.rand(node_count)

        input_weights = np.random.rand(node_count * self.dim)
        if self.include_zero_weight:
            input_weights[0] = 0
        input_weights = self.weight_min + self.weight_delta * input_weights
        input_weights = input_weights.reshape(node_count, self.dim)

        return biases, input_weights, self.random_fn_params(node_count)


class TrigBasis(AdditiveBasis):

    def __init__(self, dim, weight_range=(0, 1), bias_range=(-1, 0)):
        """ Additive function basis using a single Gaussian as non-linearity.

        Example:
        >>> fn = lambda x: np.sin(5 * x)  # want to learn this
        >>> basis = TrigBasis(1, .1)
        >>> xs = np.random.rand(100)      # 100 random points
        >>> values = fn(xs)               # then learn using 100 nodes:
        >>> params, weights = basis.extreme_learning_train(xs, values, 100)

        :param dim: Dimensionality of variable space (value space is 1D)
        :param weight_range: Range the weight can have.
        :param bias_range: Range the input bias can have.
        """
        super().__init__(dim, weight_range, bias_range)
        self.include_zero_weight = True  # equivalent to external bias

    def get_outputs(self, inputs, fn_params):
        return np.cos(inputs)

    def random_fn_params(self, node_count):
        return None
