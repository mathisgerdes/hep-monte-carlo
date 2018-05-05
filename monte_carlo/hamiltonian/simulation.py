import numpy as np


class HamiltonLeapfrog(object):

    def __init__(self, pot_gradient, kin_gradient, step_size, steps):
        """ Leapfrog method to simulate Hamiltonian propagation.

        This method is based on a general structure of the Hamiltonian of
        H = kinetic(p) + potential(q),
        where q is the "space" and p the "momentum" variable.

        :param pot_gradient: Partial derivative of the potential with
            respect to q.
        :param kin_gradient: Partial derivative of the kinetic energy
            with respect to p.
        :param step_size: Size of a simulation step in "time"-space.
        :param steps: Number of iterations to perform in each call.

        """
        self.kin_gradient = kin_gradient
        self.pot_gradient = pot_gradient
        self.step_size = step_size
        self.steps = steps

    def __call__(self, q_init, p_init):
        """ Propagate the state q, p using a given number of simulation steps.

        :param q_init: Initial space variable.
        :param p_init: Initial momentum variable.
        :return: Tuple (q_next, p_next) of state after given number of
            simulation steps.
        """
        p = np.array(p_init, copy=True, ndmin=1, subok=True)
        q = np.array(q_init, copy=True, ndmin=1, subok=True)
        pot_grad = self.pot_gradient(q)[0]
        try:
            for i in range(self.steps):
                p -= self.step_size / 2 * pot_grad
                q += self.step_size * self.kin_gradient(p)[0]
                pot_grad = self.pot_gradient(q)[0]
                p -= self.step_size / 2 * pot_grad
        except RuntimeWarning:
            # overflow, division
            return None, None
        return q, p
