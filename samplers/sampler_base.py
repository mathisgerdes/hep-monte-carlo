import numpy as np

class Sampler(object):
    def __init__(self, ndim, target_pdf, is_adaptive):
        self.ndim = ndim
        self.target_pdf = target_pdf
        self.is_adaptive = is_adaptive
    
    def step(self, current, current_pdf):
        raise NotImplementedError()

    def sample(self, nsamples, start):
        samples = np.zeros([nsamples, self.ndim])
        current = start
        current_pdf = self.target_pdf(start)
        for t in range(1, nsamples+1):
            previous = current
            previous_pdf = current_pdf
            current, current_pdf = self.step(previous, previous_pdf)
            samples[t-1] = current
            
            # try to adapt if sampler is adaptive
            if self.is_adaptive:
                self.adapt(t, current=current, current_pdf=current_pdf, previous=previous, previous_pdf=previous_pdf)
            
            if t%1000 == 0:
                print('passed: ', t, 'samples')
        
        return samples
