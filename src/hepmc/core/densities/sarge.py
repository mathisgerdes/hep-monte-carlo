from ..density import Distribution

import numpy as np


class Sarge(Distribution):

    def __init__(self, nin, nout, Ecm, pin=None, s0=None):
        super().__init__(nout * 4)
        self.sarge = SingleSarge(nin, nout, Ecm)
        self.nout = nout
        self.s0 = s0

        if pin is not None:
            pin = np.array(pin).flatten()
            pin1 = Mom4D(pin[0:4])
            pin2 = Mom4D(pin[4:8])
            pin = [pin1, pin2]
        else:
            pin1 = Mom4D([Ecm / 2, 0., 0., Ecm / 2])
            pin2 = Mom4D([Ecm / 2, 0., 0., -Ecm / 2])
            pin = [pin1, pin2]
        self.pin = pin

        if s0 is None:
            pT_min = 5.
            angle_min = 0.3
            # cut on inv. mass
            s0 = 2 * pT_min ** 2 * min(
                1 - np.cos(angle_min),
                1. / (1 + np.sqrt(1 - pT_min ** 2 / Ecm ** 2)))
        self.s0 = s0

    def proposal(self, state=None):
        point = self.sarge.generate_point(self.pin, self.s0)
        return np.concatenate([np.array(mom4._arr) for mom4 in point])

    def proposal_pdf(self, state, candidate):
        out_state = candidate.reshape(self.nout, 4)
        pout = [Vec4D(momentum) for momentum in out_state]
        return 1 / self.sarge.generate_weight(self.pin, pout, self.s0)

    def pdf(self, xs):
        return np.array([self.proposal_pdf(None, x) for x in xs])

    def pdf_gradient(self, xs):
        raise NotImplementedError

    def rvs(self, sample_count):
        return np.stack([self.proposal() for _ in range(sample_count)])


class SingleSarge(object):

    def __init__(self, nin, nout, Ecms):
        self.nin = nin
        self.nout = nout
        self.Ecms = Ecms
        self.n_xi = 2 * nout - 4

    def generate_weight(self, pin, pout, s0):
        sump = np.sum(pin)

        s = sump * sump
        xi_min = s / s0 - ((self.nout + 1.) * (self.nout - 2.)) / 2.

        weight = 2. * np.pi ** (self.nout - 1) * np.log(xi_min) ** (
            self.n_xi) * (self.n_xi + 1) / (s * s)

        for j in range(self.nout - 1):
            weight *= pout[j] * pout[j + 1]

        weight *= pout[self.nout - 1] * pout[0]
        return weight

    def generate_point(self, pin, s0):
        sump = Mom4D()

        for i in range(self.nin):
            sump += pin[i]

        ET = sump.m
        xi_min = ET * ET / s0 - ((self.nout + 1.) * (self.nout - 2.)) / 2.

        q = [Mom4D(), Mom4D()]
        costheta = 2. * np.random.random() - 1.
        sintheta = np.sqrt(1. - costheta * costheta)
        phi = 2. * np.pi * np.random.random()
        q[0] = ET / 2. * Mom4D(
            [1., sintheta * np.sin(phi), sintheta * np.cos(phi), costheta])
        q[1] = ET / 2. * Mom4D(
            [1., -sintheta * np.sin(phi), -sintheta * np.cos(phi), -costheta])
        p = self.qcd_antenna(q, xi_min)
        return p

    def qcd_antenna(self, p, xi_min):
        q = [Mom4D() for i in range(self.nout)]
        q[0] = p[0]
        q[self.nout - 1] = p[1]
        x = self.polytope(self.n_xi)
        xi = np.empty(2)
        lab = p[0] + p[1]
        logm = np.log(xi_min)
        for j in range(self.nout - 2):
            phi = 2. * np.pi * np.random.random()
            xi[0] = np.exp((x[2 * j + 1] - x[2 * j]) * logm)
            xi[1] = np.exp((x[2 * j + 2] - x[2 * j]) * logm)
            q[j + 1] = self.basic_antenna(q[j], q[self.nout - 1], xi, phi)
            lab += q[j + 1]

        Elab = lab.m
        B = -lab.mom3d / Elab
        G = lab.E / Elab
        A = 1. / (1. + G)
        scale = self.Ecms / Elab

        for j in range(self.nout):
            e = q[j].E
            BQ = np.dot(B, q[j].mom3d)
            q[j][:] = scale * np.concatenate(
                ([G * e + BQ], q[j].mom3d + B * (e + A * BQ)))

        return q

    def basic_antenna(self, pin0, pin1, xi, phi):
        cms = pin0 + pin1

        ref = Mom4D([1., 0., 0., 1.])
        pin0_cms = self.boost_inv(cms, pin0)
        E1 = pin0_cms.E

        rot = self.rotat(pin0_cms, ref)

        k0_cms = E1 * (xi[0] + xi[1])
        cos_cm = (xi[1] - xi[0]) / (xi[1] + xi[0])

        sin_cm = np.sqrt(1. - cos_cm * cos_cm)
        k_help = k0_cms * Mom4D(
            [1., sin_cm * np.sin(phi), sin_cm * np.cos(phi), cos_cm])
        k_cms = self.rotat_inv(k_help, rot)
        k = self.boost(cms, k_cms)

        return k

    def polytope(self, m):
        # Produces a uniform random distribution inside a polytope with
        # | x_k | < 1, | x_k-x_l | < 1, see physics/0003078

        x = np.empty(m + 1)

        # number of negative values
        k = int((m + 1) * np.random.random())
        x[0] = 0.

        if k == 0:
            for i in range(1, m + 1):
                x[i] = np.random.random()

        elif k == m:
            for i in range(1, m + 1):
                x[i] = -np.random.random()

        else:
            prod = 1.
            for i in range(1, k + 1):
                prod *= np.random.random()

            v1 = -np.log(prod)
            prod = 1.
            for i in range(1, m - k + 2):
                prod *= np.random.random()

            v2 = -np.log(prod)
            y1 = v1 / (v1 + v2)
            x[1] = -y1
            x[m] = (1 - y1) * np.random.random() ** (1. / (m - k))
            for i in range(2, k + 1):
                x[i] = x[1] * np.random.random()

            for i in range(k + 1, m):
                x[i] = x[m] * np.random.random()

        return np.random.permutation(x)

    def boost(self, q, ph):
        #                                      _
        # Boost of a 4-vector ( relative speed q/q(0) ):
        #
        # ph is the 4-vector in the rest frame of q
        # p is the corresponding 4-vector in the lab frame
        #
        # INPUT     OUTPUT
        # q, ph     p

        p = Mom4D()

        rsq = q.m

        p.E = (q.E * ph.E + np.dot(q.mom3d, ph.mom3d)) / rsq
        c1 = (ph.E + p.E) / (rsq + q.E)
        p.mom3d = ph.mom3d + c1 * q.mom3d

        return p

    def boost_inv(self, q, p):
        #                                      _
        # Boost of a 4-vector ( relative speed q/q(0) ):
        #
        # ph is the 4-vector in the rest frame of q
        # p is the corresponding 4-vector in the lab frame
        #
        # INPUT     OUTPUT
        # q, p      ph

        ph = Mom4D()
        rsq = q.m

        ph.E = q * p / rsq
        c1 = (p.E + ph.E) / (rsq + q.E)
        ph.mom3d = p.mom3d - c1 * q.mom3d

        return ph

    def rotat(self, p1, p2):
        # Rotation of a 4-vector:
        #
        #            p1 = rot*p2
        #
        # INPUT     OUTPUT
        #
        # p1, p2    rot

        rot = np.empty((3, 3))

        r = np.empty((2, 3, 3))
        pm = np.empty(2)
        sp = np.empty(2)
        cp = np.empty(2)
        st = np.empty(2)
        ct = np.empty(2)
        pp = [Mom4D(), Mom4D]

        pm[0] = np.sqrt(p1.mom3d.dot(p1.mom3d))
        pm[1] = np.sqrt(p2.mom3d.dot(p2.mom3d))
        pp[0] = (1. / pm[0]) * p1
        pp[1] = (1. / pm[1]) * p2

        for i in range(2):
            ct[i] = pp[i][3]
            st[i] = np.sqrt(1. - ct[i] * ct[i])
            if np.isclose(abs(ct[i]), 1.):
                cp[i] = 1.
                sp[i] = 0.
            else:
                cp[i] = pp[i][2] / st[i]
                sp[i] = pp[i][1] / st[i]

            r[i, 0, 0] = cp[i]
            r[i, 0, 1] = sp[i] * ct[i]
            r[i, 0, 2] = st[i] * sp[i]
            r[i, 1, 0] = -sp[i]
            r[i, 1, 1] = ct[i] * cp[i]
            r[i, 1, 2] = cp[i] * st[i]
            r[i, 2, 0] = 0.
            r[i, 2, 1] = -st[i]
            r[i, 2, 2] = ct[i]

            for i in range(3):
                for l in range(3):
                    rot[i, l] = 0.
                    for k in range(3):
                        rot[i, l] += r[0, i, k] * r[1, l, k]

        return rot

    def rotat_inv(self, p2, rot):
        # Rotation of a 4-vector:
        #
        #            p1 = rot*p2
        #
        # INPUT     OUTPUT
        #
        # p2, rot   p1

        p1 = Mom4D()

        p1.E = p2.E
        for i in range(3):
            p1[i + 1] = 0.
            for j in range(3):
                p1[i + 1] += rot[i, j] * p2[j + 1]

        return p1


class Vec4D(object):
    def __init__(self, arr=None):
        if arr is None:
            self._arr = np.zeros(4)
        else:
            arr = np.asarray(arr)
            if arr.shape != (4,):
                raise TypeError('Wrong array size! Must have 4 entries.')

            self._arr = arr

    def __add__(self, rhs):
        if isinstance(rhs, self.__class__):
            return self.__class__(self._arr + rhs._arr)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return self.__class__(self._arr + rhs)
        else:
            return NotImplemented

    def __sub__(self, rhs):
        if isinstance(rhs, self.__class__):
            return self.__class__(self._arr - rhs._arr)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return self.__class__(self._arr - rhs)
        else:
            return NotImplemented

    def __mul__(self, rhs):
        if isinstance(rhs, self.__class__):
            return self[0]*rhs[0] - self.vec3d.dot(rhs.vec3d)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return self.__class__(self._arr * rhs)
        else:
            return NotImplemented

    def __rmul__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            return self.__class__(lhs * self._arr)
        else:
            return NotImplemented

    def __getitem__(self, key):
        return self._arr[key]

    def __setitem__(self, key, value):
        self._arr[key] = value

    @property
    def vec3d(self):
        return self[1:]

    @vec3d.setter
    def vec3d(self, arr):
        arr = np.asarray(arr)
        if arr.shape != (3,):
            raise TypeError('Wrong array size! Must have 3 entries.')

        self[1:] = arr

class Mom4D(Vec4D):
    @property
    def mom3d(self):
        return self.vec3d

    @mom3d.setter
    def mom3d(self, p):
        self.vec3d = p

    @property
    def E(self):
        return self[0]

    @E.setter
    def E(self, E):
        self[0] = E

    @property
    def m(self):
        m2 = self*self
        if m2 < 0:
            if np.isclose(0., m2, atol=1.e-7):
                return 0.
            else:
                raise ValueError('Negative Mass!')

        return np.sqrt(m2)

    @property
    def pT(self):
        return np.sqrt(self[1]*self[1]+self[2]*self[2])

# angle between two 4-momenta
def angle(p, q):
    if not (isinstance(p, Mom4D) and isinstance(q, Mom4D)):
        raise TypeError("Arguments need to be of type 'Mom4D'")

    cos_angle = (p.mom3d.dot(q.mom3d))/(p.E*q.E)
    return np.arccos(cos_angle)