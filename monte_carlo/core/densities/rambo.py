import numpy as np
from math import gamma
from ..density import Distribution
from ..util import interpret_array

class Rambo(Distribution):

    def __init__(self, nparticles, E_CM):
        ndim = 4*nparticles
        super().__init__(ndim, False)

        self.nparticles = nparticles
        self.E_CM = E_CM

    def rvs(self, sample_size):
        q = np.empty((self.nparticles, 4))
        Q = np.zeros(4)
        for i in range(self.nparticles):
            q_i = self.generate_fourvector()
            q[i] = q_i
            Q = Q + q_i
        
        M = np.sqrt(Q[0]**2 - Q[1]**2 - Q[2]**2 - Q[3]**2)
        b = -Q[1:]/M
        x = self.E_CM/M
        gamma = Q[0]/M
        a = 1./(1.+gamma)
        
        xs = np.empty((sample_size, self.ndim))
        p = np.empty((self.nparticles, 4))
        for j in range(sample_size):
            for i in range(self.nparticles):
                q_i = q[i]
                p_i = np.zeros(4)
                bdotq = b.dot(q_i[1:])
                p_i[0] = x*(gamma*q_i[0] + bdotq)
                p_i[1] = x*(q_i[1] + b[0]*q_i[0] + a*(bdotq)*b[0])
                p_i[2] = x*(q_i[2] + b[1]*q_i[0] + a*(bdotq)*b[1])
                p_i[3] = x*(q_i[3] + b[2]*q_i[0] + a*(bdotq)*b[2])
                p[i] = p_i

            xs[j] = p.flatten()

        return xs

    def pdf(self, xs):
        return (np.pi/2.)**(self.nparticles-1) * self.E_CM**(2*self.nparticles-4)/gamma(self.nparticles)/gamma(self.nparticles-1)
    
    def generate_fourvector(self):
        ran = np.random.rand(4)
        c = 2.*ran[0]-1.
        phi = 2.*np.pi*ran[1]
    
        q = np.zeros(4)
        q[0] = -np.log(ran[2]*ran[3])
        q[1] = q[0]*np.sqrt(1-c**2)*np.cos(phi)
        q[2] = q[0]*np.sqrt(1-c**2)*np.sin(phi)
        q[3] = q[0]*c
    
        return q
