class StaticSurrogate(object):
    #def __init__(self):
        
    def train(self, samples):
        raise NotImplementedError()
    
    def log_pdf_gradient(self, x):
        raise NotImplementedError()
