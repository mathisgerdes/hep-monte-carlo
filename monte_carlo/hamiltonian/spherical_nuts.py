from .spherical_hmc import DualAveragingSphericalHMC, \
    cot, arccot, SphericalHMCState
import numpy as np


def integrator(q, pot_gradient, z, trafo, jac, stepsize, nsteps):
    cumsinq = np.cumprod(np.sin(q))
    v = z / np.concatenate(([1], cumsinq[:-1]))
    du = pot_gradient(trafo(q))[0] * jac
    
    # make a half step for velocity
    v = v - stepsize/2 * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    # alternate full steps for position and momentum
    for l in range(nsteps):
        # make a full step for position
        # 1. map to augmented sphere
        x = np.concatenate((np.cos(q), [1])) * np.concatenate(([1], cumsinq))
        dx = (np.concatenate((-v*np.tan(q), [0])) +
              np.concatenate(([0], np.cumsum(v*cot(q))))) * x
        # 2. rotate on sphere
        x0 = x
        dx_nom = np.sqrt(np.sum(dx**2))
        costdx = np.cos(dx_nom*stepsize)
        sintdx = np.sin(dx_nom*stepsize)
        x = x0*costdx + dx/dx_nom*sintdx
        dx = -x0*dx_nom*sintdx + dx*costdx
        # 3. go back to hyper-rectangle
        cumx2 = np.cumsum(x**2)
        cotq = x[:-1] / np.sqrt(1-cumx2[:-1])
        q = arccot(cotq)
        q[-1] = np.pi + np.sign(x[-1])*(q[-1]-np.pi)
        v = -cotq*(dx[:-1]/x[:-1]+np.concatenate(([0], np.cumsum(x*dx)[:-2])) /
                   (1-np.concatenate(([0], cumx2[:-2]))))
        v[-1] = v[-1] * np.sign(x[-1])
        
        # make last half step for velocity
        cumsinq = np.cumprod(np.sin(q))
        du = pot_gradient(trafo(q))[0] * jac
        if l != nsteps-1:
            v = v - stepsize * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    z = v*np.concatenate(([1], cumsinq[:-1])) - \
        stepsize/2 * du/np.concatenate(([1], cumsinq[:-1]))
    
    return q, z


class SphericalNUTS(DualAveragingSphericalHMC):
    """
    No-U-Turn Sampler with Dual Averaging for Box-Type Constraints
    """
    
    def __init__(self, target_density, adapt_schedule, lim_lower=None,
                 lim_upper=None, t0=10, stepsize_bar0=1, Hbar0=0, gamma=0.05,
                 kappa=0.75, delta=0.65, Emax=1000):

        super().__init__(target_density, 1., adapt_schedule,
                         lim_lower, lim_upper, t0, stepsize_bar0,
                         Hbar0, gamma, kappa, delta)
        self.Emax = Emax
        self.alpha = None
        self.n_alpha = None
        
    def proposal(self, current):
        # initialization
        try:
            q_minus = q_plus = q = current.theta
        except AttributeError:
            current = self.init_state(current)
            q_minus = q_plus = q = current.theta

        z0 = np.random.standard_normal(self.ndim)
        u = np.random.uniform()
        z_minus = z_plus = current.momentum = z0

        j, n, s = 0, 1, 1

        while s == 1:
            v = np.random.choice([-1, 1])
            if v == -1:
                (q_minus, z_minus, _, _, q_prime, n_prime, s_prime,
                 self.alpha, self.n_alpha) = self.build_tree(
                    q_minus, z_minus, u, v, j, self.stepsize, q, z0, self.Emax)
            else:
                (_, _, q_plus, z_plus, q_prime, n_prime, s_prime, self.alpha,
                 self.n_alpha) = self.build_tree(
                    q_plus, z_plus, u, v, j, self.stepsize, q, z0, self.Emax)

            if s_prime == 1 and np.random.uniform() < min(1, n_prime/n):
                q = q_prime

            dq = q_plus - q_minus
            n = n + n_prime
            s = s_prime * (np.dot(dq, z_minus) >= 0) * (np.dot(dq, z_plus) >= 0)
            j = j + 1

        x = self.theta_to_x(q)
        prob = self.target_density.pdf(x)
        return SphericalHMCState(x, tag=self.log_weight(q), pdf=prob, theta=q)

    def build_tree(self, q, z, u, v, j, stepsize, q0, z0, Emax):
        if j == 0:
            # Base case - take one leapfrog step in the direction v.
            q_prime, z_prime = integrator(
                q, self.target_density.pot_gradient, z, self.theta_to_x,
                self.J_xtheta, v*stepsize, nsteps=1)
    
            E = -self.target_density.pot(self.theta_to_x(q_prime)) - \
                .5*np.sum(z_prime**2)
            E0 = -self.target_density.pot(self.theta_to_x(q0)) - \
                 .5*np.sum(z0**2)
            dE = E - E0
            
            n_prime = (np.log(u) - dE <= 0)
            s_prime = (np.log(u) - dE < Emax)
            return (q_prime, z_prime, q_prime, z_prime, q_prime, n_prime,
                    s_prime, min(1, np.exp(dE)), 1)
        else:
            # Recursion - implicitly build the left and right subtrees.
            (q_minus, z_minus, q_plus, z_plus, q_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = self.build_tree(
                q, z, u, v, j - 1, stepsize, q0, z0, Emax)
            if s_prime == 1:
                if v == -1:
                    (q_minus, z_minus, _, _, q_2prime, n_2prime, s_2prime,
                     alpha_2prime, n_alpha_2prime) = self.build_tree(
                        q_minus, z_minus, u, v, j - 1, stepsize, q0, z0, Emax)
                else:
                    (_, _, q_plus, z_plus, q_2prime, n_2prime, s_2prime,
                     alpha_2prime, n_alpha_2prime) = self.build_tree(
                        q_plus, z_plus, u, v, j - 1, stepsize, q0, z0, Emax)
                if np.random.uniform() < n_2prime/max(n_prime + n_2prime, 1.):
                    q_prime = q_2prime
    
                alpha_prime = alpha_prime + alpha_2prime
                n_alpha_prime = n_alpha_prime + n_alpha_2prime
    
                dq = q_plus - q_minus
                s_prime = (s_2prime * (np.dot(dq, z_minus) >= 0) *
                           (np.dot(dq, z_plus) >= 0))
                n_prime = n_prime + n_2prime
            return (q_minus, z_minus, q_plus, z_plus, q_prime, n_prime,
                    s_prime, alpha_prime, n_alpha_prime)
        
    def accept(self, state, candidate):
        return 1  # accept all

    def adapt(self, iteration, prev, current, accept):
        super().adapt(iteration, prev, current, self.alpha / self.n_alpha)

    #
    # def sample(self, nsamples, start):
    #     samples = np.zeros([nsamples, self.ndim])
    #     weights = np.zeros(nsamples)
    #     current_q = self.x_to_theta(start)
    #     samples[0] = start
    #     weights[0] = (np.arange(1, self.ndim)-self.ndim).dot(np.log(np.sin(current_q[:-1]))) + np.sum(np.log(self.J_xtheta)) # log weight
    #     for t in range(1, nsamples):
    #         current_q = self.step(current_q)
    #         # back to constrained domain
    #         x = self.theta_to_x(current_q)
    #         weight = (np.arange(1, self.ndim)-self.ndim).dot(np.log(np.sin(current_q[:-1]))) + np.sum(np.log(self.J_xtheta)) # log weight
    #         samples[t] = x
    #         weights[t] = weight
    #
    #         self.adapt(t, aprob=self.alpha/self.n_alpha)
    #
    #         if (t+1)%1000 == 0:
    #             print('passed: ', t+1, 'samples')
    #
    #     #resample
    #     weights = np.exp(weights-weights.mean())
    #     weights = weights/weights.sum()
    #     resamp_idx = np.random.choice(np.arange(nsamples), nsamples, replace=True, p=weights)
    #     samples = samples[resamp_idx]
    #
    #     return samples, samples.mean(), samples.var()
