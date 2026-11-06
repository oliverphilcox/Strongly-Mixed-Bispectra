"""Tree-level contact bispectra of the pi-sigma system with exact quadratic mixing, in units H = 1.

For a vertex -lambda (d_t^{n_1} X_1)(d_t^{n_2} X_2)(d_t^{n_3} X_3) or -lambda' a^{-2} (d_t^{n_1} Y_1)(d_i Y_2)(d^i Y_3) in
a^{-3} L_3, the shape S = (k_1 k_2 k_3)^2 B / ((2 pi)^4 Delta_zeta^4) is returned per unit lambda / Delta_zeta. The time
integral runs along real time from tau_* to 0, and along tau_* + i eta before.
"""

from itertools import permutations

import numpy as np

from .modes import gauss_legendre, NODES_PER_PANEL

Y_STAR = 10.0
LOG_X_MIN = -15.0
LOG_PANEL = 1.6
X_LIN = 0.3
PANEL = 4.0
ETA_LIN = 0.2
ETA_MAX = 200.0

VERTICES = {
    'pidot3': (('pidot', 'pidot', 'pidot'), False),
    'sigma_pidot2': (('sigma', 'pidot', 'pidot'), False),
    'pidot_sigma2': (('pidot', 'sigma', 'sigma'), False),
    'pidot_sigmadot2': (('pidot', 'sigmadot', 'sigmadot'), False),
    'sigma3': (('sigma', 'sigma', 'sigma'), False),
    'pidot_dpi2': (('pidot', 'pi', 'pi'), True),
    'sigma_dpi2': (('sigma', 'pi', 'pi'), True),
    'sigmadot_dpi_dsigma': (('sigmadot', 'pi', 'sigma'), True),
}


class Bispectrum:
    """Contact bispectra for one set of mode functions."""

    def __init__(self, modes, n_nodes=NODES_PER_PANEL):

        self.modes = modes
        c = modes.c_s
        s0 = modes.s0

        # Set up late times, x_t = -k_t tau in (0, x_*)
        edges = np.append(np.arange(LOG_X_MIN, np.log(X_LIN), LOG_PANEL), np.log(X_LIN))
        x_log, w_log = gauss_legendre(edges, n_nodes)
        x_log = np.exp(x_log)
        n_lin = int(np.ceil((Y_STAR - X_LIN) * max(c, 1 / c) / PANEL))
        x_lin, w_lin = gauss_legendre(np.linspace(X_LIN, Y_STAR, n_lin + 1), n_nodes)
        self.x_late = np.concatenate([x_log, x_lin]) / s0
        self.dx_late = np.concatenate([x_log * w_log, w_lin]) / s0
        self.n_first = len(x_log) // (len(edges) - 1)

        # Set up early times, x_t = x_* - i eta
        eta_lin, w_lin = gauss_legendre([0.0, ETA_LIN], n_nodes)
        edges = np.append(np.arange(np.log(ETA_LIN), np.log(ETA_MAX), LOG_PANEL), np.log(ETA_MAX))
        eta_log, w_log = gauss_legendre(edges, n_nodes)
        eta_log = np.exp(eta_log)
        self.x_early = (Y_STAR - 1j * np.concatenate([eta_lin, eta_log])) / s0
        self.deta = np.concatenate([w_lin, eta_log * w_log]) / s0

        self.h = modes.commutator((Y_STAR / 2 + 0.5) / s0)

    def propagators(self, e):
        """2 k^3 G at x = e x_t for the legs pi, pidot, sigma and sigmadot, along late and early times."""

        modes = self.modes
        c = modes.c_s
        h0, h1 = modes.head_integrals()
        out = {}
        for part, x in (('late', e * self.x_late), ('early', e * self.x_early)):

            # Compute moments int dt w s^n e^{-isx}, and those with (s^2 - c_s^2)
            mom = np.zeros((4, len(x)), complex)
            for s, w in modes.rays:

                phase = np.exp(-1j * np.outer(x, s)) * (modes.dt * w)
                for n in range(4):

                    mom[n] += phase @ s**n
            mom += h0 * np.exp(-1j * x)
            mom_sigma0 = mom[2] - c**2 * mom[0] + 2 * h1 * np.exp(-1j * x)
            mom_sigma1 = mom[3] - c**2 * mom[1] + 2 * h1 * np.exp(-1j * x)
            plane = modes.plane_wave * np.exp(-1j * c * x)

            # Apply the operators giving each leg
            legs = {'pi': mom[0] + 1j * x * mom[1] + plane * (1 + 1j * c * x),
                    'pidot': -x**2 * (mom[2] + c**2 * plane),
                    'sigma': x**2 * mom_sigma0 / modes.rho,
                    'sigmadot': -x**2 * (2 * mom_sigma0 - 1j * x * mom_sigma1) / modes.rho}

            # Replace imaginary parts by the commutator along real time
            if part == 'late':
                h, dh, d2h, d3h = self.h(x)
                im = {'pi': -(h - x * dh) / c**2,
                      'pidot': -x**2 * d2h / c**2,
                      'sigma': x**2 * (d2h + c**2 * h) / (modes.rho * c**2),
                      'sigmadot': -x**2 * (2 * (d2h + c**2 * h) + x * (d3h + c**2 * dh)) / (modes.rho * c**2)}
                legs = {k: v.real + 1j * im[k] for k, v in legs.items()}
            out[part] = legs
        return out

    def shapes(self, k1, k2, k3, names=tuple(VERTICES)):
        """Shapes of the requested vertices at one triangle."""

        modes = self.modes
        e = np.array([k1, k2, k3], float) / (k1 + k2 + k3)
        legs = [self.propagators(ej) for ej in e]
        norm = np.prod(e)**2 * modes.c_s**4.5 / (np.pi * modes.R**1.5)
        result = {}
        for name in names:

            fields, gradient = VERTICES[name]
            power = 2 if gradient else 4

            # Continue the late-time scaling x_t^gamma below the first node
            n_sigma_pidot = sum(f != 'pi' for f in fields)
            gamma = 2 * gradient - 1 + (n_sigma_pidot - 1) * (1.5 - modes.nu_eff)
            dx = (self.dx_late * self.x_late**(-power)).astype(complex)
            dx[:self.n_first] /= -np.expm1(-LOG_PANEL * (gamma + 1))
            deta = self.deta * self.x_early**(-power)

            # Sum over assignments of legs to momenta
            total = 0j
            for perm in permutations(range(3)):

                late = np.prod([legs[j]['late'][f] / (2 * e[j]**3) for f, j in zip(fields, perm)], axis=0)
                early = np.prod([legs[j]['early'][f] / (2 * e[j]**3) for f, j in zip(fields, perm)], axis=0)
                integral = np.sum(dx * late.imag) - np.sum(deta * early).real
                if gradient:
                    a, b = perm[1], perm[2]
                    integral = -integral * (e[3 - a - b]**2 - e[a]**2 - e[b]**2) / 2
                total += integral
            result[name] = float(-norm * total.real)
        return result
