from samplers.mcmc.hmc import leapfrog, DualAveragingHMC
import numpy as np

class NUTS(DualAveragingHMC):
    """
    No-U-Turn Sampler with Dual Averaging
    """
    
    def __init__(self, ndim, target_pdf, target_log_pdf_gradient, start, adapt_schedule, t0=10, stepsize_bar0=1, Hbar0=0, gamma=0.05, kappa=0.75, delta=0.65, Emax=1000, momentum=None):
        super().__init__(ndim, target_pdf, target_log_pdf_gradient, 1, start, adapt_schedule, t0, stepsize_bar0, Hbar0, gamma, kappa, delta, momentum)
        self.Emax = Emax
        self.alpha = None
        self.n_alpha = None
    
    def proposal(self, current, current_pdf):
        p0 = self.momentum.rvs()
        #u = np.random.uniform()
        u = np.random.uniform(0, current_pdf/self.momentum.pdf(p0))
        print('u_max:', current_pdf/self.momentum.pdf(p0))
        q_minus, q_plus, p_minus, p_plus, q = current, current, p0, p0, current
        j, n, s = 0, 1, 1

        while s == 1:
            v = np.random.choice([-1, 1])
            if v == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime, self.alpha, self.n_alpha = self.buildtree(q_minus, p_minus, u, v, j, self.stepsize, q, p0, self.Emax)
            else:
                _, _, q_plus, p_plus, q_prime, n_prime, s_prime, self.alpha, self.n_alpha = self.buildtree(q_plus, p_plus, u, v, j, self.stepsize, q, p0, self.Emax)

            if s_prime == 1 and np.random.uniform() < min(1, n_prime/n):
                q = q_prime

            dq = q_plus - q_minus
            n = n + n_prime
            s = s_prime * (np.dot(dq, p_minus) >= 0) * (np.dot(dq, p_plus) >= 0)
            j = j + 1

        q_pdf = self.target_pdf(q)
        return q, q_pdf
    
    def buildtree(self, q, p, u, v, j, stepsize, q0, p0, Emax):
        if j == 0:
            # Base case - take one leapfrog step in the direction v.
            q_prime, p_prime = leapfrog(q, self.target_log_pdf_gradient, p, self.momentum.log_pdf_gradient, v*stepsize, nsteps=1)
    
            dE = self.target_pdf(q_prime)/self.momentum.pdf(p_prime)
            print('dE:',dE)
            dE0 = self.target_pdf(q0)/self.momentum.pdf(p0)
            print('dE0:',dE0)
            print('u:', u)
            n_prime = (u <= dE)
            s_prime = (u < Emax*dE)
            return q_prime, p_prime, q_prime, p_prime, q_prime, n_prime, s_prime, min(1, dE/dE0), 1
        else:
            print('RECURSION!')
            # Recursion - implicitly build the left and right subtrees.
            q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = self.buildtree(q, p, u, v, j-1, stepsize, q0, p0, Emax)
            if s_prime == 1:
                if v == -1:
                    q_minus, p_minus, _, _, q_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = self.buildtree(q_minus, p_minus, u, v, j-1, stepsize, q0, p0, Emax)
                else:
                    _, _, q_plus, p_plus, q_2prime, n_2prime, s_2prime, alpha_2prime, n_alpha_2prime = self.buildtree(q_plus, p_plus, u, v, j-1, stepsize, q0, p0, Emax)
                if np.random.uniform() < n_2prime/(n_prime + n_2prime):
                    q_prime = q_2prime
    
                alpha_prime = alpha_prime + alpha_2prime
                n_alpha_prime = n_alpha_prime + n_alpha_2prime
    
                dq = q_plus - q_minus
                s_prime = s_2prime * (np.dot(dq, p_minus) >= 0) * (np.dot(dq, p_plus) >= 0)
                n_prime = n_prime + n_2prime
            return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        
    def aprob(self, current_q_pdf, proposal_q_pdf, current_p_pdf, proposal_p_pdf):
        raise NotImplementedError()
    
    def step(self, current_q, current_q_pdf):
        return self.proposal(current_q, current_q_pdf)
    
    def sample(self, nsamples, start):
        samples = []
        current_q = start
        current_q_pdf = self.target_pdf(start)
        for t in range(1, nsamples+1):
            print(t)
            current_q, current_q_pdf = self.step(current_q, current_q_pdf)
            samples.append(current_q)
            print('alpha:', self.alpha)
            print('n_alpha:', self.n_alpha)
            self.adapt(t, current_q, current_q_pdf, aprob=self.alpha/self.n_alpha)
            print('stepsize:', self.stepsize)
        
        return samples
