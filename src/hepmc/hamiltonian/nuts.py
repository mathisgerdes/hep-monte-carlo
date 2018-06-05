import numpy as np

from .dual_average import DualAveragingHMC
from ..core.markov.metropolis import MetropolisState


class NUTSUpdate(DualAveragingHMC):
    """
    No-U-Turn Sampler with Dual Averaging
    """
    
    def __init__(self, target_density, p_dist, adapt_schedule,
                 t0=10, step_size_bar0=1, Hbar0=0, gamma=0.05, kappa=0.75,
                 delta=0.65, Emax=1000):
        super().__init__(target_density, p_dist, 1, adapt_schedule,
                         t0, step_size_bar0, Hbar0, gamma, kappa, delta)
        self.Emax = Emax
        self.alpha = None
        self.n_alpha = None
    
    def proposal(self, current):
        p0 = self.p_dist.proposal()
        # u = np.random.uniform()
        u = np.random.uniform(0, current.pdf / self.p_dist.pdf(p0))
        # print('u_max:', current.pdf / self.p_dist.pdf(p0))
        q_minus, q_plus, p_minus, p_plus, q = current, current, p0, p0, current
        j, n, s = 0, 1, 1

        while s == 1:
            v = np.random.choice([-1, 1])
            if v == -1:
                (q_minus, p_minus, _, _, q_prime, n_prime, s_prime, self.alpha,
                 self.n_alpha) = self.build_tree(q_minus, p_minus, u, v, j,
                                                 self.step_size, q, p0,
                                                 self.Emax)
            else:
                (_, _, q_plus, p_plus, q_prime, n_prime, s_prime, self.alpha,
                 self.n_alpha) = self.build_tree(q_plus, p_plus, u, v, j,
                                                 self.step_size, q, p0,
                                                 self.Emax)

            if s_prime == 1 and np.random.uniform() < min(1, n_prime/n):
                q = q_prime

            dq = q_plus - q_minus
            n = n + n_prime
            s = s_prime * (np.dot(dq, p_minus) >= 0) * (np.dot(dq, p_plus) >= 0)
            j = j + 1

        q_pdf = self.target_density.pdf(q)
        return MetropolisState(q, pdf=q_pdf)
    
    def build_tree(self, q, p, u, v, j, step_size, q0, p0, Emax):
        if j == 0:
            # Base case - take one leapfrog step in the direction v.
            q_prime, p_prime = self.simulate_custom(q, p, 1, v * step_size)
    
            dE = self.target_density.pdf(q_prime)/self.p_dist.pdf(p_prime)
            # print('dE:',dE)
            dE0 = self.target_density.pdf(q0)/self.p_dist.pdf(p0)
            # print('dE0:',dE0)
            # print('u:', u)
            n_prime = (u <= dE)
            s_prime = (u < Emax*dE)
            return (q_prime, p_prime, q_prime, p_prime, q_prime, n_prime,
                    s_prime, min(1, dE/dE0), 1)
        else:
            # print('RECURSION!')
            # Recursion - implicitly build the left and right subtrees.
            (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = self.build_tree(
                    q, p, u, v, j - 1, step_size, q0, p0, Emax)

            if s_prime == 1:
                if v == -1:
                    (q_minus, p_minus, _, _, q_2prime, n_2prime, s_2prime,
                     alpha_2prime, n_alpha_2prime) = self.build_tree(
                        q_minus, p_minus, u, v, j - 1, step_size, q0, p0, Emax)
                else:
                    (_, _, q_plus, p_plus, q_2prime, n_2prime, s_2prime,
                     alpha_2prime, n_alpha_2prime) = self.build_tree(
                        q_plus, p_plus, u, v, j - 1, step_size, q0, p0, Emax)
                if np.random.uniform() < n_2prime/(n_prime + n_2prime):
                    q_prime = q_2prime
    
                alpha_prime = alpha_prime + alpha_2prime
                n_alpha_prime = n_alpha_prime + n_alpha_2prime
    
                dq = q_plus - q_minus
                s_prime = (s_2prime * (np.dot(dq, p_minus) >= 0) *
                           (np.dot(dq, p_plus) >= 0))
                n_prime = n_prime + n_2prime
            return (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime,
                    s_prime, alpha_prime, n_alpha_prime)
        
    def accept(self, state, candidate):
        return 1  # accept all
