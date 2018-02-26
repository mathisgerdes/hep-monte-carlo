class Sampler(object):
    def __init__(self, ndim, target_pdf, is_adaptive):
        self.ndim = ndim
        self.target_pdf = target_pdf
        self.is_adaptive = is_adaptive
    
    def step(self, current, current_pdf):
        raise NotImplementedError()

    def sample(self, nsamples, start):
        samples = [start]
        pdf = self.target_pdf(start)
        previous_pdf = None
        for t in range(1, nsamples+1):
            previous_pdf = pdf
            value, pdf = self.step(samples[-1], pdf)
            samples.append(value)
            
            # try to adapt if sampler is adaptive
            if self.is_adaptive:
                self.adapt(t, current=value, current_pdf=pdf, previous=samples[-1], previous_pdf=previous_pdf)
            
            if t%1000 == 0:
                print('passed: ', t, 'samples')
        
        return samples[1:]
