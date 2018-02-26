from samplers.mcmc.mcmc_base import MCMC
from proposals.gaussian import IsotropicZeroMeanGaussian
import numpy as np

def leapfrog(q, dlogq, p, dlogp, stepsize, nsteps):
    # half momentum update
    q_log_pdf_gradient = dlogq(q)
    if q_log_pdf_gradient is not np.inf:
        p = p - (stepsize / 2) * -q_log_pdf_gradient
    else:
        return np.nan, np.nan
    
    # alternate full variable and momentum updates
    for i in range(nsteps):
        p_log_pdf_gradient = dlogp(p)
        if p_log_pdf_gradient is not np.inf:
            q = q + stepsize * -p_log_pdf_gradient
        else:
            return np.nan, np.nan

        # precompute since used for two half-steps
        try:
            dlogq_eval = dlogq(q)
        except RuntimeWarning:
            return np.nan, np.nan
            
        if dlogq_eval is not np.inf:
            #  first half momentum update
            p = p - (stepsize / 2) * -dlogq_eval
            
            # second half momentum update
            if i != nsteps - 1:
                p = p - (stepsize / 2) * -dlogq_eval
        else:
            return np.nan, np.nan

    return q, p

class StaticHMC(MCMC):
    """
    Hamiltonian Monte Carlo samplers
    """
    
    def __init__(self, ndim, target_pdf, target_log_pdf_gradient, stepsize_min, stepsize_max, nsteps_min, nsteps_max, is_adaptive=False, momentum=None):
        if momentum is None:
            momentum = IsotropicZeroMeanGaussian(ndim)

        momentum.is_symmetric = False
        super().__init__(ndim, target_pdf, momentum, is_adaptive)
        
        self.target_log_pdf_gradient = target_log_pdf_gradient
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.nsteps_min = nsteps_min
        self.nsteps_max = nsteps_max
        self.momentum = momentum
        
    
    def proposal(self, current, current_pdf):
        # sample momentum and leapfrog parameters
        p0 = self.momentum.rvs()
        p0_pdf = self.momentum.pdf(p0)
        nsteps = np.random.randint(self.nsteps_min, self.nsteps_max + 1)
        stepsize = np.random.rand() * (self.stepsize_max - self.stepsize_min) + self.stepsize_min
        
        q, p = leapfrog(current, self.target_log_pdf_gradient, p0, self.momentum.log_pdf_gradient, stepsize, nsteps)
        if np.isnan([q, p]).any():
            return np.nan, np.nan, np.nan, np.nan, p0, p0_pdf
        
        try:
            q_pdf = self.target_pdf(q)
            p_pdf = self.momentum.pdf(p)
        except RuntimeWarning:
            q_pdf = np.nan
            p_pdf = np.nan
        
        return q, q_pdf, p, p_pdf, p0, p0_pdf

    def aprob(self, current_q_pdf, proposal_q_pdf, current_p_pdf, proposal_p_pdf):
        #return min(1, proposal_q_pdf*current_p_pdf / (current_q_pdf*proposal_p_pdf))
        #return min(1, current_q_pdf*proposal_p_pdf / (proposal_q_pdf*current_p_pdf))
        if np.isnan([current_q_pdf, proposal_q_pdf, current_p_pdf, proposal_p_pdf]).any():
            return 0.
        else:
            try:
                return min(1, proposal_q_pdf*proposal_p_pdf / (current_q_pdf*current_p_pdf))
            except RuntimeWarning:
                return 0.
            #return min(1, proposal_q_pdf*current_p_pdf / (current_q_pdf*proposal_p_pdf))
    
    def step(self, current_q, current_q_pdf):
        proposal_q, proposal_q_pdf, proposal_p, proposal_p_pdf, current_p, current_p_pdf = self.proposal(current, current_pdf)
        aprob = self.aprob(current_q_pdf, proposal_q_pdf, current_p_pdf, proposal_p_pdf)
        
        if aprob > np.random.uniform():
            return proposal_q, proposal_q_pdf, proposal_p, proposal_p_pdf
        
        else:
            return current_q, current_q_pdf, current_p, current_p_pdf
        
    def sample(self, nsamples, start):
        samples = []
        current_q = start
        current_q_pdf = self.target_pdf(start)
        for t in range(1, nsamples+1):
            proposal_q, proposal_q_pdf, proposal_p, proposal_p_pdf, current_p, current_p_pdf = self.proposal(current_q, current_q_pdf)
            aprob = self.aprob(current_q_pdf, proposal_q_pdf, current_p_pdf, proposal_p_pdf)
            
            if aprob > np.random.uniform():
                samples.append(proposal_q)
                current_q = proposal_q
                current_q_pdf = proposal_q_pdf
            else:
                samples.append(current_q)
            
            # try to adapt if sampler is adaptive
            if self.is_adaptive:
                self.adapt(t, current_q, current_q_pdf, aprob)
            
            if t%1000 == 0:
                print('passed: ', t, 'samples')
        
        return samples

class DualAveragingHMC(StaticHMC):
    """
    Adapts the stepsize by dual averaging
    """
    
    def __init__(self, ndim, target_pdf, target_log_pdf_gradient, simulation_length, start, adapt_schedule, t0=10, stepsize_bar0=1, Hbar0=0, gamma=0.05, kappa=0.75, delta=0.65, momentum=None):
        is_adaptive = True
        super().__init__(ndim, target_pdf, target_log_pdf_gradient, None, None, None, None, is_adaptive, momentum)
        self.simulation_length = simulation_length
        self.adapt_schedule = adapt_schedule
        self.t0 =t0
        self.stepsize_bar = stepsize_bar0
        self.Hbar = Hbar0
        self.gamma = gamma
        self.kappa = kappa
        self.delta = delta
        start_pdf = target_pdf(start)
        self.stepsize = self.stepsize_min = self.stepsize_max = self.find_reasonable_stepsize(start, start_pdf)        
        print('stepsize:', self.stepsize)
        self.nsteps_min = self.nsteps_max = int(simulation_length / self.stepsize)
        self.mu = np.log(10*self.stepsize)
        
    def find_reasonable_stepsize(self, current, current_pdf):
        stepsize = .25 / self.ndim**(1/4)
        print('current:', current)
        print('current_pdf:', current_pdf)
        print('stepsize:', stepsize)
        p0 = self.momentum.rvs()
        print('p0:', p0)
        p0_pdf = self.momentum.pdf(p0)
        print('p0_pdf:', p0_pdf)
        q, p = leapfrog(current, self.target_log_pdf_gradient, p0, self.momentum.log_pdf_gradient, stepsize, nsteps=1)
        print('q:', q)
        print('p:', p)
        q_pdf = self.target_pdf(q)
        try:
            p_pdf = self.momentum.pdf(p)
            aprob = q_pdf*p_pdf/(current_pdf*p0_pdf)
        except RuntimeWarning:
            aprob = 0
        #print('q_pdf:', q_pdf)
        #print('p_pdf:', p_pdf)
        #    
        #while np.isnan(q).any() or np.isnan(p).any() or q_pdf == 0 or p_pdf == 0:
        #    p0 = self.momentum.rvs()
        #    print('p0:', p0)
        #    p0_pdf = self.momentum.pdf(p0)
        #    print('p0_pdf:', p0_pdf)
        #    q, p = leapfrog(current, self.target_log_pdf_gradient, p0, self.momentum.log_pdf_gradient, stepsize, nsteps=1)
        #    print('q:', q)
        #    print('p:', p)
        #    q_pdf = self.target_pdf(q)
        #    try:
        #        p_pdf = self.momentum.pdf(p)
        #    except RuntimeWarning:
        #        p_pdf = 0
        #    print('q_pdf:', q_pdf)
        #    print('p_pdf:', p_pdf)
            
        #a = 2. * (self.aprob(current_pdf, q_pdf, p0_pdf, p_pdf) > 0.5) - 1.
        
        #while (q_pdf*self.momentum.pdf(current)/(current_pdf*self.momentum.pdf(q)))**a > 2**(-a):
        #while (q_pdf*p_pdf/(current_pdf*p0_pdf))**a > 2**(-a):
        print('aprob:', aprob)
        a = 2. * (aprob > 0.5) - 1.
        print('a', a)
        while aprob == 0. or aprob**a > 2**(-a):
            stepsize = 2.**a * stepsize
            print('stepsize:', stepsize)
            q, p = leapfrog(current, self.target_log_pdf_gradient, p0, self.momentum.log_pdf_gradient, stepsize, nsteps=1)
            q_pdf = self.target_pdf(q)
            try:
                p_pdf = self.momentum.pdf(p)
                aprob = q_pdf*p_pdf/(current_pdf*p0_pdf)
            except RuntimeWarning:
                aprob = 0
            print('aprob:', aprob)
        
        return stepsize
    
    def adapt(self, t, current, current_pdf, aprob):
        if self.adapt_schedule(t) is True:
            self.Hbar = (1 - 1/(t+self.t0)) * self.Hbar + 1/(t+self.t0) * (self.delta - aprob)
            log_stepsize = self.mu - np.sqrt(t)/self.gamma*self.Hbar
            self.stepsize = np.exp(log_stepsize)
            if self.stepsize >= self.simulation_length:
                self.stepsize = self.simulation_length
                self.log_stepsize = np.log(self.stepsize)

            self.stepsize_bar = np.exp(t**(-self.kappa) * log_stepsize + (1-t**(-self.kappa))*np.log(self.stepsize_bar))
            self.nsteps_min = self.nsteps_max = int(self.simulation_length / self.stepsize)
        
        else:
            self.stepsize = self.stepsize_bar
        
        print('aprob:', aprob)
        print('stepsize:', self.stepsize)
        print('nsteps:', self.nsteps_min)
