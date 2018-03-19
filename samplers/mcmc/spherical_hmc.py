from samplers.mcmc.mcmc_base import MCMC
import numpy as np

def cot(x):
    if (x==0).any():
        return np.inf
    else:
        return 1/np.tan(x)
    
def arccot(x):
    return np.pi/2 - np.arctan(x)

def integrator(q, dlogq, z, du, trafo, jac, stepsize, nsteps):
    cumsinq = np.cumprod(np.sin(q))
    v = z / np.concatenate(([1], cumsinq[:-1]))
    
    # make a half step for velocity
    v = v - stepsize/2 * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    # alternate full steps for position and momentum
    for l in range(nsteps):
        # make a full step for position
        # 1. map to augmented sphere
        x = np.concatenate((np.cos(q), [1])) * np.concatenate(([1], cumsinq))
        dx = (np.concatenate((-v*np.tan(q), [0])) + np.concatenate(([0], np.cumsum(v*cot(q))))) * x
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
        v = -cotq*(dx[:-1]/x[:-1]+np.concatenate(([0], np.cumsum(x*dx)[:-2]))/(1-np.concatenate(([0], cumx2[:-2]))))
        v[-1] = v[-1] * np.sign(x[-1])
        
        # make last half step for velocity
        cumsinq = np.cumprod(np.sin(q))
        du = dlogq(trafo(q))*jac
        if l != nsteps-1:
            v = v - stepsize * du/np.concatenate(([1], cumsinq[:-1]**2))
    
    z = v*np.concatenate(([1], cumsinq[:-1])) - stepsize/2 * du/np.concatenate(([1], cumsinq[:-1]))
    
    return q, z, du

class StaticSphericalHMC(MCMC):
    """
    Spherical HMC in spherical coordinates with unit radius
    for box type constraints
    """
    
    def __init__(self, ndim, target_log_pdf, target_log_pdf_gradient, stepsize_min, stepsize_max, nsteps_min, nsteps_max, lim_lower=None, lim_upper=None, is_adaptive=False):
        super().__init__(ndim, target_log_pdf, proposal_dist=None, is_adaptive=is_adaptive)
        
        # default limits: unit hypercube
        if lim_lower is None:
            lim_lower = np.zeros(ndim)
        if lim_upper is None:
            lim_upper = np.ones(ndim)
        
        self.target_log_pdf = target_log_pdf
        self.target_log_pdf_gradient = target_log_pdf_gradient
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.nsteps_min = nsteps_min
        self.nsteps_max = nsteps_max
        self.lim_lower = lim_lower
        self.lim_upper = lim_upper
        
        self.J_xtheta = (self.lim_upper-self.lim_lower)/(np.concatenate((np.full(self.ndim-1, 1), [2]))*np.pi)
    
    def theta_to_x(self, theta):
        return self.lim_lower + theta*self.J_xtheta
    
    def x_to_theta(self, x):
        return (x-self.lim_lower) / self.J_xtheta
    
    def proposal(self, current_q, current_target_log_pdf, current_target_log_pdf_gradient):
        # initialization
        q = current_q
        u = current_target_log_pdf
        du = current_target_log_pdf_gradient
        
        # sample velocity
        z = np.random.standard_normal(self.ndim)
        
        # evaluate energy at start of trajectory
        E_cur = u + .5*np.sum(z**2)
        
        # sample integrator parameters
        nsteps = np.random.randint(self.nsteps_min, self.nsteps_max + 1)
        stepsize = np.random.rand() * (self.stepsize_max - self.stepsize_min) + self.stepsize_min
        
        # integrate
        q, z, du = integrator(q, self.target_log_pdf_gradient, z, du, self.theta_to_x, self.J_xtheta, stepsize, nsteps)
        
        # evaluate energy at the end of the trajectory
        u = self.target_log_pdf(self.theta_to_x(q))
        E_prp = u + .5*np.sum(z**2)

        return q, z, u, du, E_cur, E_prp

    def aprob(self, current_H, proposal_H):
        return min(1, np.exp(-proposal_H+current_H))
    
    def step(self, current_q, current_target_log_pdf, current_target_log_pdf_gradient):
        proposal_q, proposal_p, proposal_target_log_pdf, proposal_target_log_pdf_gradient, current_H, proposal_H = self.proposal(current_q, current_target_log_pdf, current_target_log_pdf_gradient)
        aprob = self.aprob(current_H, proposal_H)
        
        if aprob > np.random.uniform():
            return proposal_q, proposal_target_log_pdf, proposal_target_log_pdf_gradient
        
        else:
            return current_q, current_target_log_pdf, current_target_log_pdf_gradient
        
    def sample(self, nsamples, start):
        samples = np.zeros([nsamples, self.ndim])
        weights = np.zeros(nsamples)
        nacc_tot = 0 # monitor acceptance probability over all samples
        nacc_batch = 0 # ... and per batch
        batch_length = 1000
        
        current_q = self.x_to_theta(start)
        samples[0] = start
        weights[0] = (np.arange(1, self.ndim)-self.ndim).dot(np.log(np.sin(current_q[:-1]))) + np.sum(np.log(self.J_xtheta)) # log weight
        current_target_log_pdf = self.target_log_pdf(start)
        current_target_log_pdf_gradient = self.target_log_pdf_gradient(start)
        for t in range(1, nsamples):
            proposal_q, proposal_p, proposal_target_log_pdf, proposal_target_log_pdf_gradient, current_H, proposal_H = self.proposal(current_q, current_target_log_pdf, current_target_log_pdf_gradient)
            aprob = self.aprob(current_H, proposal_H)
            
            x = self.theta_to_x(proposal_q)
            print('x:', x)
            
            if aprob > np.random.uniform():
                nacc_tot += 1
                nacc_batch += 1
                print('accept')
                # back to constrained domain
                x = self.theta_to_x(proposal_q)
                weight = (np.arange(1, self.ndim)-self.ndim).dot(np.log(np.sin(proposal_q[:-1]))) + np.sum(np.log(self.J_xtheta)) # log weight
                samples[t] = x
                weights[t] = weight
                
                current_q = proposal_q
                current_target_log_pdf = proposal_target_log_pdf
                current_target_log_pdf_gradient = proposal_target_log_pdf_gradient
            else:
                print('reject')
                samples[t] = samples[t-1]
                weights[t] = weights[t-1]
            
            # try to adapt if sampler is is_adaptive
            if self.is_adaptive:
                self.adapt(t, aprob)
            
            if (t+1)%batch_length == 0:
                print('passed:', t+1, 'samples (aprob =', nacc_batch/batch_length, ')')
                nacc_batch = 0.
        
        #resample
        weights = np.exp(weights-weights.mean())
        weights = weights/weights.sum()
        resamp_idx = np.random.choice(np.arange(nsamples), nsamples, replace=True, p=weights)
        samples = samples[resamp_idx]
        
        return samples, samples.mean(), samples.var(), nacc_tot

class DualAveragingSphericalHMC(StaticSphericalHMC):
    """
    Adapts the stepsize by dual averaging
    """
    
    def __init__(self, ndim, target_log_pdf, target_log_pdf_gradient, simulation_length, start, adapt_schedule, lim_lower=None, lim_upper=None, t0=10, stepsize_bar0=1, Hbar0=0, gamma=0.05, kappa=0.75, delta=0.65):
        is_adaptive = True
        super().__init__(ndim, target_log_pdf, target_log_pdf_gradient, None, None, None, None, lim_lower, lim_upper, is_adaptive)
        
        self.simulation_length = simulation_length
        self.adapt_schedule = adapt_schedule
        self.t0 = t0
        self.stepsize_bar = stepsize_bar0
        self.Hbar = Hbar0
        self.gamma = gamma
        self.kappa = kappa
        self.delta = delta
        self.stepsize = self.stepsize_min = self.stepsize_max = self.find_reasonable_stepsize(self.x_to_theta(start), self.target_log_pdf(start), self.target_log_pdf_gradient(start))        
        print('stepsize:', self.stepsize)
        self.nsteps_min = self.nsteps_max = int(simulation_length / self.stepsize)
        self.mu = np.log(10*self.stepsize)
        
    def find_reasonable_stepsize(self, current_q, current_target_log_pdf, current_target_log_pdf_gradient):
        stepsize = 1.
        
        # initialization
        current_u = current_target_log_pdf
        current_du = current_target_log_pdf_gradient
        
        # sample velocity
        current_z = np.random.standard_normal(self.ndim)
        
        # evaluate energy at start of trajectory
        E_cur = current_u + .5*np.sum(current_z**2)
        
        # integrate one step
        proposal_q, proposal_z, proposal_du = integrator(current_q, self.target_log_pdf_gradient, current_z, current_du, self.theta_to_x, self.J_xtheta, stepsize, nsteps=1)
        
        # evaluate energy at the end of the trajectory
        proposal_u = self.target_log_pdf(self.theta_to_x(proposal_q))
        E_prp = proposal_u + .5*np.sum(proposal_z**2)

        aprob = self.aprob(E_cur, E_prp)
        print('aprob:', aprob)
            
        a = 2. * (aprob > 0.5) - 1.
        print('a', a)
        while aprob**a > 2**(-a):
            stepsize = 2.**a * stepsize
            print('stepsize:', stepsize)
            proposal_q, proposal_z, proposal_du = integrator(current_q, self.target_log_pdf_gradient, current_z, current_du, self.theta_to_x, self.J_xtheta, stepsize, nsteps=1)
            proposal_u = self.target_log_pdf(self.theta_to_x(proposal_q))
            E_prp = proposal_u + .5*np.sum(proposal_z**2)
            aprob = self.aprob(E_cur, E_prp)
            print('aprob:', aprob)
        
        # limit the stepsize to reasonable values
        min_stepsize = 1e-3
        return max(stepsize, min_stepsize)
    
    def adapt(self, t, aprob):
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
            if self.stepsize >= self.simulation_length:
                self.stepsize = self.simulation_length
                
            self.nsteps_min = self.nsteps_max = int(self.simulation_length / self.stepsize)
        
        #print('aprob:', aprob)
        #print('stepsize:', self.stepsize)
        #print('nsteps:', self.nsteps_min)
