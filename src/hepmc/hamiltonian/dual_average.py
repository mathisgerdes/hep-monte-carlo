import numpy as np
from .hmc import HamiltonianUpdate


class DualAveragingHMC(HamiltonianUpdate):
    """
    Adapts the stepsize by dual averaging
    """

    def __init__(self, target_density, p_dist, simulation_length,
                 adapt_schedule, t0=10, stepsize_bar0=1, Hbar0=0, gamma=0.05,
                 kappa=0.75, delta=0.65):
        # steps and step size are set later
        super().__init__(target_density, p_dist, 1, 1., is_adaptive=True)
        # set later
        self.step_size_min = self.step_size_max = self.step_size
        self.nsteps_max = None
        self.nsteps_min = None
        self.mu = None

        self.simulation_length = simulation_length
        self.adapt_schedule = adapt_schedule
        self.t0 = t0
        self.step_size_bar = stepsize_bar0
        self.Hbar = Hbar0
        self.gamma = gamma
        self.kappa = kappa
        self.delta = delta

    def init_adapt(self, initial_state):
        self.step_size = self.find_reasonable_step_size(initial_state)
        self.step_size_min = self.step_size_max = self.step_size
        # print('stepsize: ' + str(self.step_size))

        self.nsteps_max = int(self.simulation_length / self.step_size)
        self.nsteps_min = self.nsteps_max
        self.mu = np.log(10 * self.step_size)

    def simulate_custom(self, current, p0, steps, step_size):
        old_step_size, old_steps = self.step_size, self.steps
        self.step_size = step_size
        self.steps = steps
        q, p = self.simulate(current, p0)
        self.step_size, self.steps = old_step_size, old_steps
        return q, p

    def find_reasonable_step_size(self, current):
        step_size = .25 / self.ndim ** (1 / 4)
        # print('current:', current)
        # print('current_pdf:', current.pdf)
        # print('stepsize:', step_size)
        p0 = self.p_dist.proposal()
        # print('p0:', p0)
        p0_pdf = self.p_dist.pdf(p0)
        # print('p0_pdf:', p0_pdf)

        q, p = self.simulate_custom(current, p0, 1, step_size)

        # print('q:', q)
        # print('p:', p)
        q_pdf = self.target_density.pdf(q)
        try:
            p_pdf = self.p_dist.pdf(p)
            aprob = q_pdf * p_pdf / (current.pdf * p0_pdf)
        except RuntimeWarning:
            aprob = 0
        # print('q_pdf:', q_pdf)
        # print('p_pdf:', p_pdf)
        #    
        # while (np.isnan(q).any() or np.isnan(p).any() or
        #        q_pdf == 0 or p_pdf == 0):
        #    p0 = self.momentum.rvs()
        #    print('p0:', p0)
        #    p0_pdf = self.momentum.pdf(p0)
        #    print('p0_pdf:', p0_pdf)
        #    q, p = self.simulate_custom(current, p0, 1, step_size)
        #    print('q:', q)
        #    print('p:', p)
        #    q_pdf = self.target_pdf(q)
        #    try:
        #        p_pdf = self.momentum.pdf(p)
        #    except RuntimeWarning:
        #        p_pdf = 0
        #    print('q_pdf:', q_pdf)
        #    print('p_pdf:', p_pdf)

        # a = 2. * (self.aprob(current_pdf, q_pdf, p0_pdf, p_pdf) > 0.5) - 1.

        # while (q_pdf*self.momentum.pdf(current)/
        #        (current_pdf*self.momentum.pdf(q)))**a > 2**(-a):
        # while (q_pdf*p_pdf/(current_pdf*p0_pdf))**a > 2**(-a):
        # print('aprob:', aprob)
        a = 2. * (aprob > 0.5) - 1.
        # print('a', a)
        while aprob == 0. or aprob ** a > 2 ** (-a):
            step_size = 2. ** a * step_size
            # print('stepsize:', step_size)
            q, p = self.simulate_custom(current, p0, 1, step_size)
            q_pdf = self.target_density.pdf(q)
            try:
                p_pdf = self.p_dist.pdf(p)
                aprob = q_pdf * p_pdf / (current.pdf * p0_pdf)
            except RuntimeWarning:
                aprob = 0
            # print('aprob:', aprob)

        return step_size

    def adapt(self, t, prev, current, accept):
        if self.adapt_schedule(t):
            self.Hbar = (1 - 1 / (t + self.t0)) * self.Hbar + 1 / (
                         t + self.t0) * (self.delta - accept)
            log_step_size = self.mu - np.sqrt(t) / self.gamma * self.Hbar
            self.step_size = np.exp(log_step_size)
            if self.step_size >= self.simulation_length:
                self.step_size = self.simulation_length
                log_step_size = np.log(self.step_size)

            self.step_size_bar = np.exp(t ** (-self.kappa) * log_step_size + (
                        1 - t ** (-self.kappa)) * np.log(self.step_size_bar))
            self.nsteps_min = self.nsteps_max = int(
                self.simulation_length / self.step_size)

        else:
            self.step_size = self.step_size_bar

        # print('aprob:', accept)
        # print('stepsize:', self.step_size)
        # print('nsteps:', self.nsteps_min)
