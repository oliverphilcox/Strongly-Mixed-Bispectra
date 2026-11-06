"""Exact linear mode functions of the pi-sigma system with quadratic mixing, in units H = 1."""

import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial
from numpy.polynomial import chebyshev as Ch
from scipy.integrate import solve_ivp

LOG_T_EDGES = np.arange(-30.0, 26.0, 1.5)
NODES_PER_PANEL = 14
X_SERIES = 0.3
N_SERIES = 60
PANEL_H = 0.25
N_CHEB = 18


def gauss_legendre(edges, n=NODES_PER_PANEL):
    """Gauss-Legendre nodes and weights on consecutive panels."""

    x, w = np.polynomial.legendre.leggauss(n)
    nodes = []
    weights = []
    for a, b in zip(edges[:-1], edges[1:]):

        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


class Modes:
    """Mode functions at sound-speed ratio c_s, mixing rho and effective mass m_eff, with m_eff^2 = m^2 + rho^2 > 0.

    Along the conjugate rays s = s_alpha - it from the light cones s = c_s and s = 1,

        sum_alpha pi_alpha(0) F_alpha^*(x) = plane_wave e^{-i c_s x} + sum_alpha int dt w_alpha(s) e^{-isx}.
    """

    def __init__(self, c_s, rho, m_eff, n_nodes=NODES_PER_PANEL):

        self.c_s = float(c_s)
        self.rho = float(rho)
        self.m_eff = float(m_eff)
        self.m2 = self.m_eff**2 - self.rho**2
        self.nu_eff = np.sqrt(2.25 - self.m_eff**2 + 0j)
        self.s0 = min(self.c_s, 1.0)

        # Set up nodes along the rays
        log_t, w_log = gauss_legendre(LOG_T_EDGES, n_nodes)
        self.t = 2 * self.s0 * np.exp(log_t)
        self.dt = w_log * self.t
        self.t_min = 2 * self.s0 * np.exp(LOG_T_EDGES[0])
        self.t_max = 2 * self.s0 * np.exp(LOG_T_EDGES[-1])

        # Compute weights and late-time amplitudes
        if self.c_s == 1:
            self._equal_speeds()
        else:
            self._unequal_speeds()
        self.R = self.c_s**3 * np.sum(np.abs(self.pi0)**2)

    def _heun(self):
        """Coefficients of p2 g'' + p1 g' + p0 g = 0."""

        c2 = self.c_s**2
        p2 = np.array([c2, 0.0, -(1 + c2), 0.0, 1.0])
        p1 = np.array([0.0, -2 * (2 + c2), 0.0, 6.0])
        p0 = np.array([-2 + (2 - self.m2) * c2, 0.0, self.m2 + self.rho**2 + 4])
        return p2, p1, p0

    def _taylor(self, r, n_max=80):
        """Taylor coefficients of the solution regular at s = r with g(r) = 1, in powers of (s - r) / scale."""

        scale = min(abs(r - o) for o in (0.0, 1.0, -1.0, self.c_s, -self.c_s) if o != r)
        p2, p1, p0 = (np.concatenate([Polynomial(p)(Polynomial([r, scale])).coef, np.zeros(n_max + 3)]) for p in self._heun())
        p2 /= scale**2
        p1 /= scale
        p2[0] = 0.0
        a = np.zeros(n_max + 1)
        a[0] = 1.0
        for m in range(n_max):

            n = m - np.arange(m + 1)
            rest = ((p2[2:m + 3] * n * (n - 1) + p1[1:m + 2] * n) * a[n]).sum() + (p0[:m + 1] * a[m::-1]).sum()
            a[m + 1] = -rest / ((m + 1) * (p2[1] * m + p1[0]))
        return a, scale

    @staticmethod
    def _taylor_eval(a, scale, ds):
        """(g, g') at a distance ds from the cone."""

        powers = (ds / scale)**np.arange(len(a))
        g = np.sum(a * powers)
        dg = np.sum(a[1:] * np.arange(1, len(a)) * powers[:-1]) / scale
        return np.array([g, dg], complex)

    def _step(self, y, s_from, s_to):
        """Transport (G, G'), with G = (s^2 - c_s^2) g, along a straight segment using the Heun equation."""

        c2 = self.c_s**2
        q = 2 - self.m_eff**2
        ds = s_to - s_from

        def rhs(u, yy):
            """First-order form along the segment."""

            s = s_from + u * ds
            return ds * np.array([yy[1], (2 * s * yy[1] - (q + self.rho**2 * c2 / (c2 - s**2)) * yy[0]) / (1 - s**2)])

        return solve_ivp(rhs, (0.0, 1.0), np.asarray(y, complex), method='DOP853', rtol=1e-12, atol=1e-16).y[:, -1]

    def _ray(self, r, s):
        """The solution regular at s = r with g(r) = 1, at the ordered nodes s of a ray leaving r."""

        a, scale = self._taylor(r)
        far = np.abs(s - r) > 0.2 * scale
        g = np.zeros(len(s), complex)
        g[~far] = [self._taylor_eval(a, scale, si - r)[0] for si in s[~far]]

        # Transport beyond the Taylor disc
        far = np.flatnonzero(far)
        s_prev = r + 0.2 * scale * (s[far[0]] - r) / abs(s[far[0]] - r)
        g0, dg0 = self._taylor_eval(a, scale, s_prev - r)
        y = np.array([(s_prev**2 - self.c_s**2) * g0, 2 * s_prev * g0 + (s_prev**2 - self.c_s**2) * dg0])
        for i in far:

            y = self._step(y, s_prev, s[i])
            s_prev = s[i]
            g[i] = y[0] / (s[i]**2 - self.c_s**2)
        return g

    def _late_time(self, s, g):
        """int dt g(s_min - it), with the tail g ~ s^{nu_eff - 5/2} beyond the last node."""

        p = self.nu_eff - 2.5
        s_edge = s[0] - 1j * (self.t_max - self.t[0])
        tail = -1j * g[-1] * s[-1]**(-p) * s_edge**(p + 1) / (p + 1)
        return np.sum(self.dt * g) + tail

    def _unequal_speeds(self):
        """g_pi from s = c_s and g_sigma from s = 1."""

        c = self.c_s
        A1 = 1j * self.rho**2 / (2 * np.sqrt(c) * (1 - c**2))
        B2 = -1j * self.rho / (1 - c**2)
        s_pi = c - 1j * self.t
        s_sigma = 1 - 1j * self.t
        g_pi = self._ray(c, s_pi)
        g_sigma = self._ray(1.0, s_sigma)

        # Compute late-time amplitudes
        pi1 = 1j / c**1.5 + 1j * A1 * np.conj(self._late_time(s_pi, g_pi))
        pi2 = 1j * B2 * np.conj(self._late_time(s_sigma, g_sigma))
        self.pi0 = np.array([pi1, pi2])

        # Compute weights on the conjugate rays
        self.rays = [(s_pi, pi1 * np.conj(1j * A1) * g_pi), (s_sigma, pi2 * np.conj(1j * B2) * g_sigma)]
        self.plane_wave = pi1 * np.conj(1j / c**1.5)
        self.heads = []

    def _equal_speeds(self):
        """g_+ and g_- from s = 1, in closed form."""

        with mp.workdps(20):

            nu = mp.sqrt(mp.mpf('2.25') - mp.mpf(self.m_eff)**2)
            p = {+1: mp.mpc(0, self.rho / 2), -1: mp.mpc(0, -self.rho / 2)}

            def g(a, s):
                """Legendre solution g_a(s)."""

                return ((s - 1)**(p[a] - 1) * ((s + 1) / 2)**(-p[a] - 1)
                        * mp.hyp2f1(mp.mpf('0.5') - nu, mp.mpf('0.5') + nu, 1 + 2 * p[a], (1 - s) / 2))

            def pi_late(a):
                """Late-time amplitude pi_a(0)."""

                num = mp.gamma(mp.mpf('0.5') + p[a]) * mp.gamma(mp.mpf('0.75') - nu / 2) * mp.gamma(mp.mpf('0.75') + nu / 2)
                den = mp.sqrt(mp.pi) * mp.gamma(mp.mpf('0.75') - nu / 2 + p[a]) * mp.gamma(mp.mpf('0.75') + nu / 2 + p[a])
                return 1j / mp.sqrt(2) * mp.exp(-1j * mp.pi * p[a] / 2) * 2**p[a] * num / den

            # Compute late-time amplitudes
            C1 = 1j * mp.exp(-1j * mp.pi * p[+1] / 2) / (mp.sqrt(2) * mp.gamma(p[+1]))
            D2 = 1j * mp.exp(-1j * mp.pi * p[-1] / 2) / (mp.sqrt(2) * mp.gamma(p[-1]))
            pi1 = pi_late(+1)
            pi2 = pi_late(-1)

            # Compute weights on the conjugate ray, using conj g_a(1 + it) = g_{-a}(1 - it)
            weight = {-1: pi1 * mp.conj(1j * C1), +1: pi2 * mp.conj(1j * D2)}
            s_ray = 1 - 1j * self.t
            w = np.array([complex(sum(weight[a] * g(a, mp.mpc(si.real, si.imag)) for a in (-1, +1))) for si in s_ray])
            self.heads = [(complex(weight[a]), complex(p[a])) for a in (-1, +1)]

        self.pi0 = np.array([complex(pi1), complex(pi2)])
        self.rays = [(s_ray, w)]
        self.plane_wave = 0.0

    def head_integrals(self):
        """Integrals of w and w (s - 1)/(-it) over 0 < t < t_min (c_s = 1), continued from Re p > 0."""

        h0 = 0j
        h1 = 0j
        for weight, p in self.heads:

            h0 += weight * np.exp(-0.5j * np.pi * (p - 1)) * self.t_min**p / p
            h1 += weight * np.exp(-0.5j * np.pi * p) * self.t_min**(p + 1) / (p + 1)
        return h0, h1

    def commutator(self, x_max):
        """h(x) and its first three derivatives, where h = x + O(x^3) is odd and Im G^F = -h / (2 k^3 c_s^2)."""

        c2 = self.c_s**2
        m_eff2 = self.m_eff**2
        x0 = X_SERIES / max(self.c_s, 1.0)
        powers = np.arange(N_SERIES) + 1.0

        # Compute the power series
        a = np.zeros(N_SERIES)
        a[0] = 1.0
        for n in range(2, N_SERIES, 2):

            k = n + 1.0
            a[n] = -(((1 + c2) * (k - 2) * (k - 3) + 2 * c2 * (k - 2) + c2 * (self.m2 - 2)) * a[n - 2]
                     + (c2 * a[n - 4] if n >= 4 else 0.0)) / (k * (k - 1) * (k * k - 3 * k + m_eff2))
        series = [a, a * powers, a * powers * (powers - 1), a * powers * (powers - 1) * (powers - 2)]

        def h_series(x, k):
            """k-th derivative of h from the power series."""

            return (series[k] * x[..., None]**(powers - k)).sum(-1)

        def rhs(x, y):
            """Fourth-order equation in first-order form."""

            return [y[1], y[2], y[3], -(2 * x * y[3] + ((1 + c2) * x * x + m_eff2 - 2) * y[2] + 2 * c2 * x * y[1]
                                        + c2 * (x * x + self.m2 - 2) * y[0]) / (x * x)]

        # Integrate beyond the series and fit Chebyshev panels
        y0 = [h_series(np.array([x0]), k)[0] for k in range(4)]
        sol = solve_ivp(rhs, (x0, x_max), y0, method='DOP853', rtol=1e-12, atol=1e-300, dense_output=True)
        edges = np.append(np.arange(x0, x_max, PANEL_H / max(self.c_s, 1.0)), x_max)
        mid = (edges[1:] + edges[:-1]) / 2
        half = (edges[1:] - edges[:-1]) / 2
        xk = np.cos(np.pi * (np.arange(N_CHEB) + 0.5) / N_CHEB)
        values = sol.sol((mid[:, None] + half[:, None] * xk).ravel()).reshape(4, len(mid), N_CHEB)
        cheb = Ch.chebfit(xk, values.transpose(2, 0, 1).reshape(N_CHEB, -1), N_CHEB - 1).reshape(N_CHEB, 4, len(mid))

        def h(x):
            """(h, h', h'', h''') at real 0 <= x <= x_max."""

            x = np.asarray(x, float)
            out = np.empty((4, len(x)))
            near = x < x0
            out[:, near] = [h_series(x[near], k) for k in range(4)]
            panel = np.minimum(np.searchsorted(edges, x[~near], side='right') - 1, len(mid) - 1)
            basis = Ch.chebvander((x[~near] - mid[panel]) / half[panel], N_CHEB - 1)
            out[:, ~near] = np.einsum('ik,kji->ji', basis, cheb[:, :, panel])
            return out

        return h
