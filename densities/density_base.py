from numpy import log, inf, seterr
#seterr(all='warn')
import warnings
#warnings.filterwarnings('error')

class Density(object):
    def pdf(self, x):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        return log(self.pdf(x))
    
    def pdf_gradient(self, x):
        raise NotImplementedError()
    
    def log_pdf_gradient(self, x):
        try:
            pdf = self.pdf(x)
        except RuntimeWarning:
            return inf
            
        if pdf != 0.:
            try:
                return self.pdf_gradient(x) / pdf
            except RuntimeWarning:
                return inf
        else:
            return inf
