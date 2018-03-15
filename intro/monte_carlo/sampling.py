import numpy as np
import matplotlib.pyplot as plt

def assure_2d(array):
    if array.ndim == 2:
        return array
    elif array.ndim == 1:
        return array[:, np.newaxis]
    else:
        raise RuntimeError("Array must be 1 or 2 dimensional.")
            
# ACCEPTION REJECTION
class AcceptionRejection(object):
    # for use in monte carlo it will be useful to have a parameter-less sample function, 
    # therefore use a callable object
    def __init__(self, p, C, dim=1, sample_pdf=lambda *x: 1, sample_generator=None):
        """
        Args:
            p: Unnormalized (!) probability distribution function.
            sample_pdf: Probability distribution of known sample_generator.
        """
        self.p = p
        self.C = C
        self.dim = dim
        self.sample_pdf = sample_pdf
        if sample_generator is None:
            sample_generator = lambda N: np.random.rand(N * self.dim).reshape(N, self.dim)
        self.sample = sample_generator
        
    def __call__(self, N):
        x = np.empty((N, self.dim))
            
        indices = np.arange(N)
        while indices.size > 0:
            proposal = assure_2d(self.sample(indices.size))
            accept = np.random.rand(indices.size) * self.C * self.sample_pdf(*proposal.transpose()) <= self.p(*proposal.transpose())
            x[indices[accept]] = proposal[accept]
            indices = indices[np.logical_not(accept)]
        return x
    
    
# METROPOLIS MARKOV CHAINS
class Metropolis(object):
    def __init__(self, initial, pdf, dim=1, proposal_generator=None):
        """
        Note, here the proposal must not depend on the current state. Otherwise use 
        the Metropolis Hasting algorithm.
        """
        if proposal_generator is None:
            proposal_generator = lambda s: np.random.rand(dim)
        
        self.state = initial
        self.pdf = pdf
        self.dim = dim
        self.proposal_generator = proposal_generator
        
    def __call__(self, N=1, get_accept_rate=False):
        chain = np.empty((N, self.dim))
        
        if get_accept_rate:
            accepted = 0
            
        for i in range(N):
            proposal = self.proposal_generator(self.state)
            # hasting ratio
            r = self.pdf(proposal)/self.pdf(self.state)
            a = min(1, r)
            if a == 1 or np.random.rand() < a:
                self.state = chain[i] = proposal
                
                if get_accept_rate:
                    accepted += 1
            else:
                chain[i] = self.state
                
        if get_accept_rate:
            return chain, accepted / N
        return chain
    
    
class MetropolisHasting(object):
    def __init__(self, initial, pdf, dim=1, proposal_pdf=None, proposal_generator=None):
        """
        Note, here the proposal must not depend on the current state. Otherwise use 
        the Metropolis Hasting algorithm.
        
        Args:
            proposal_pdf: takes two dim-dimensional numpy arrays (state, candidate)
        """
        if proposal_pdf is None or proposal_generator is None:
            proposal_pdf = lambda state, candidate: 1
            proposal_generator = lambda s: np.random.rand(dim)
        
        self.state = initial
        self.pdf = pdf
        self.dim = dim
        self.proposal_pdf = proposal_pdf
        self.proposal_generator = proposal_generator
        
    def __call__(self, N=1, get_accept_rate=False):
        chain = np.empty((N, self.dim))
        
        if get_accept_rate:
            accepted = 0
        
        for i in range(N):
            proposal = self.proposal_generator(self.state)
            # hasting ratio
            r = self.pdf(proposal)*self.proposal_pdf(proposal, self.state)/self.pdf(self.state)/self.proposal_pdf(self.state, proposal)
            a = min(1, r)
            if a == 1 or np.random.rand() < a:
                self.state = chain[i] = proposal
                
                if get_accept_rate:
                    accepted += 1
            else:
                chain[i] = self.state
        
        if get_accept_rate:
            return chain, accepted / N
        return chain
    
# CHANNELS for Monte Carlo Multi-Channel
class Channels(object):
    def __init__(self, sampling_channels, channel_pdfs, cweights=None):
        assert len(sampling_channels) == len(channel_pdfs), "Need a pdf for each sampling function."
        
        self.count = len(sampling_channels)
        self.cweights = np.ones(self.count)/self.count if cweights is None else cweights
        self.init_cweights = self.cweights
        self.sampling_channels = sampling_channels
        self.channel_pdfs = channel_pdfs
        
        self.Ni = np.zeros(self.count)  # hold sample_count
        
    def reset(self):
        self.cweights = self.init_cweights
        
    def pdf(self, *x):
        p = 0
        for i in range(self.count):
            p += self.cweights[i] * self.channel_pdfs[i](*x)
        return p
    
    def plot_pdf(self):
        x = np.linspace(0, 1, 1000)
        y = [self.pdf(xi) for xi in x]
        plt.plot(x, y, label="total pdf")
        
    def sample(self, N, return_sizes=False):
        choice = np.random.rand(N)
        sample_sizes = np.empty(self.count, dtype=np.int)     
        a_cum = 0
        for i in range(self.count):
            sample_sizes[i] = np.count_nonzero(np.logical_and(a_cum < choice, choice <= a_cum + self.cweights[i]))
            a_cum += self.cweights[i]
        
        sample_channel_indices = np.where(sample_sizes > 0)[0]
        
        samples = np.concatenate([assure_2d(self.sampling_channels[i](sample_sizes[i])) 
                                  for i in sample_channel_indices])
        
        if return_sizes:
            return samples, sample_sizes
        else:
            return samples
        
    def generate_sample(self, N):            
        samples, sample_sizes = self.sample(N, True)
        
        self.full_sample_sizes = sample_sizes
        # ignore channels with sample_count == 0
        self.sample_channel_indices = np.where(sample_sizes > 0)[0]
        self.sample_cweights = self.cweights[self.sample_channel_indices]
        self.sample_sizes = self.full_sample_sizes[self.sample_channel_indices]
        self.sample_count = self.sample_sizes.size  # number of channels active in current sample
        self.sample_bounds = np.array([np.sum(self.sample_sizes[0:i]) for i in range(self.sample_count)])
        
        self.samples = samples
        self.sample_weights = self.pdf(*samples.transpose())
        
    def update_sample_cweights(self, new_cweights):
        """ Update channels at self.sample_channel_indices with given values. """
        self.cweights[self.sample_channel_indices] = new_cweights

        
# VOLUMES for Stratified Sampling
# For the strafield monte carlo varient we first need a way to encode the volumes, 
# then iterate over them and sample each one appropriately.
# The simplest division is simple cubic
class CubeVolumes(object):
    def __init__(self, divisions:int, Ns={}, otherNs=1, dim=1):
        """ Initialize a division of a <dim>-dimensional hypercube into <divisions> cubes along each dimension.
        
        divisions: number of subregions along one axis. For dim=3, there will be divisions^3 subregions.
        Ns: dictionary for setting the number of samples for specific subregions (cubes).
            Example: Ns={(1,0,0): 4} to assign 4 samples to the cube 1 away from the origion along the first dimension.
        otherNs: number of samples to be assigned to all remaining cubes.
        dim: dimensionality of the cube.
        """
        self.Ns = Ns
        self.divisions = divisions
        self.dim = dim
        # volume of all regions is constant
        self.vol = 1/divisions**dim
        self.otherNs = otherNs
        self.totalN = sum(Ns.values()) + (divisions**dim - len(Ns)) * otherNs
        
    def plot_pdf(self, label="sampling weights"):
        # visualization of 1d volumes
        assert self.dim == 1, "Can only plot volumes in 1 dimension."
        x = [self.vol*i for i in range(self.divisions)]
        height = [N/self.totalN/self.vol for N,_,_ in self.iterate()] # bar height corresponds to pdf
        rects1 = plt.bar(x, height, self.vol, align='edge', alpha=.4, label=label)
    
    def iterate(self, multiple=1):
        """ multiply N in each division by multiple. Total number of samples is self.totalN()*multiple. 
        
        yields tuple (N, sample, vol), 
        where N is the number of samples in the region, sample is a function that samples the region uniformly.
        """
        for index in np.ndindex(*([self.divisions]*self.dim)):
            def sample_region():
                return np.array(index) / self.divisions + np.random.rand(self.dim) / self.divisions
            yield multiple * (self.Ns[index] if index in self.Ns else self.otherNs), sample_region, self.vol