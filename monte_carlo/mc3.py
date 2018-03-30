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

    def integrate(self, Ns_integration):
        self.integral, self.integral_var = self.mc_importance(self.fn, *Ns_integration)
        return self.integral, self.integral_var

    def sample(self, N_sample, beta):
        sample = np.empty((N_sample, self.dim))
        for i in range(N_sample):
            if np.random.rand() <= beta:
                self.sample_METROPOLIS.state = sample[i] = self.sample_IS(1)
            else:
                self.sample_IS.state = sample[i] = self.sample_METROPOLIS(1)

        return sample

    def __call__(self, Ns_integration, N_sample, beta):
        self.integrate(Ns_integration)
        return self.sample(N_sample, beta)
