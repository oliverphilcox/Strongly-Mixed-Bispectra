"""Kernel table of FastShapes for unequal sound speeds, c_pi = c != 1, c_sigma = 1.

With s = c(1+2u), u_sigma = (1-c)/(2c), and h_c, h_1 the solutions of the Heun spectral equation analytic at s = c
and s = 1 (h_c(c) = h_1(1) = 1), continued along the real axis from above,
    Omega_pi    = delta_+(u) + lam^2 c^2/(1-c^2) (1+u) Theta(u) h_c(s+i0),
    Omega_sigma = 2 lam c^{5/2}/(1-c^2) (1+u) Theta(u-u_sigma) h_1(s+i0),
    Im Omega_X = sinh(chi) Omega_Y,    (X, Y) = (pi, sigma) for c < 1, (sigma, pi) for c > 1,
    omega_+ = (Re Omega_X + i cosh(chi) Omega_Y) / sqrt(cosh chi),    r_+ = int omega_+/(1+u),    R = cosh(chi) |r_+|^2.
For c > 1 the table is in u' = c u + u_pi, u_pi = (c-1)/2, with Omega' = Omega (1+u')/(1+u'+u_pi) and
    omega'_+ = (Omega'_sigma + i e^{-chi} Omega'_pi) / sqrt(cosh chi),    Omega'_sigma on a semicircle above u' = u_pi.
Extra table keys: 'w' (int f du = sum w u f), 'omega_V' (sigma-leg weight), 'chi', 'c', 'v' = min(c, 1).
"""
import numpy as np
from numpy.polynomial import Polynomial, polynomial as Pn
from scipy.integrate import solve_ivp
import FastShapes as fs


def _polys(c, M2, lam):
    """Coefficients of P2, P1, P0 in the spectral equation P2 h'' + P1 h' + P0 h = 0."""
    return (np.array([c**2, 0.0, -(1 + c**2), 0.0, 1.0]), np.array([0.0, -2 * (2 + c**2), 0.0, 6.0]),
            np.array([-2 + (2 - M2) * c**2, 0.0, M2 + lam**2 + 4]))


def _taylor(c, M2, lam, r, nmax=80):
    """Coefficients a_n of the solution analytic at s = r, h = sum_n a_n ((s-r)/dist)^n, and the scale dist."""
    dist = min(abs(r - o) for o in (0.0, 1.0, -1.0, c, -c) if o != r)
    p2, p1, p0 = (np.concatenate([Polynomial(p)(Polynomial([r, dist])).coef, np.zeros(nmax + 3)])
                  for p in _polys(c, M2, lam))
    p2 /= dist**2; p1 /= dist; p2[0] = 0.0
    a = np.zeros(nmax + 1); a[0] = 1.0
    for m in range(nmax):
        n = m - np.arange(m + 1)
        rest = ((p2[2:m + 3] * n * (n - 1) + p1[1:m + 2] * n) * a[n]).sum() + (p0[:m + 1] * a[m::-1]).sum()
        a[m + 1] = -rest / ((m + 1) * (p2[1] * m + p1[0]))
    return a, dist


def _eval(a, dist, sig):
    """(h, h') at s = r + sig from the Taylor coefficients of _taylor."""
    pw = (sig / dist) ** np.arange(len(a))
    return np.array([np.sum(a * pw), np.sum(a[1:] * np.arange(1, len(a)) * pw[:-1]) / dist], complex)


def _transport(P, y, s_from, s_to):
    """Integrate (h, h') of the spectral equation along the straight segment s_from -> s_to."""
    P2, P1, P0 = P
    ds = s_to - s_from
    f = lambda t, yy: ds * np.array([yy[1], -(Pn.polyval(s_from + t * ds, P1) * yy[1]
                                              + Pn.polyval(s_from + t * ds, P0) * yy[0]) / Pn.polyval(s_from + t * ds, P2)])
    return solve_ivp(f, (0.0, 1.0), np.asarray(y, complex), method='DOP853', rtol=1e-12, atol=1e-16).y[:, -1]


def _h_on_nodes(c, M2, lam, s):
    """h_in(s), h_out(s) at the nodes s: the solutions analytic at min(c,1) and max(c,1), the first continued above."""
    P = _polys(c, M2, lam)
    r_in, r_out = min(c, 1.0), max(c, 1.0)
    a_in, d_in = _taylor(c, M2, lam, r_in); a_out, d_out = _taylor(c, M2, lam, r_out)
    t_in, t_out = 0.2 * d_in, 0.2 * d_out
    arc = np.flatnonzero(s.imag > 0); arc = arc[np.argsort(-np.angle(s[arc] - r_out))]
    rd = abs(s[arc[0]] - r_out) if len(arc) else 0.5 * min(r_out - r_in, 1.0)
    hin, hout = np.zeros(len(s), complex), np.zeros(len(s), complex)
    order = [i for i in np.argsort(s.real) if s[i].imag == 0]
    sel = lambda lo, hi: [i for i in order if lo < s[i].real <= hi]

    def march(y, s0, idx, h):
        for i in idx:
            y = _transport(P, y, s0, s[i]); s0 = s[i]; h[i] = y[0]
        return y, s0

    for i in sel(r_in, r_in + t_in):
        hin[i] = _eval(a_in, d_in, s[i] - r_in)[0]
    y, s0 = march(_eval(a_in, d_in, t_in), r_in + t_in, sel(r_in + t_in, r_out - rd), hin)
    y = _transport(P, y, s0, r_out - rd)
    march(y, r_out - rd, sel(r_out - rd, r_out), hin)
    if len(arc):
        y, s0 = march(y, r_out - rd, arc, hin)
        y = _transport(P, y, s0, r_out + rd)
    else:
        phis = np.linspace(np.pi, 0.0, 33)
        for phi0, phi1 in zip(phis[:-1], phis[1:]):
            y = _transport(P, y, r_out + rd * np.exp(1j * phi0), r_out + rd * np.exp(1j * phi1))
    march(y, r_out + rd, sel(r_out, r_out + rd)[::-1], hin)
    march(y, r_out + rd, sel(r_out + rd, np.inf), hin)
    for i in sel(r_out, r_out + t_out):
        hout[i] = _eval(a_out, d_out, s[i] - r_out)[0]
    march(_eval(a_out, d_out, t_out), r_out + t_out, sel(r_out + t_out, np.inf), hout)
    return hin, hout


def _grid(u_sigma):
    """Nodes u and weights w (int f du = sum w u f) of FastShapes' log-u panel grid, graded towards u_sigma."""
    edges = list(fs._T_EDGES)
    ts = np.log(u_sigma)
    if edges[0] < ts < edges[-1]:
        extra = [ts] + [ts + sgn * fs._T_PANEL * 2.0**(-k) for k in range(18) for sgn in (-1, 1)]
        edges = sorted(set(edges + [e for e in extra if edges[0] < e < edges[-1]]))
    t, w = fs._panel_gauss_legendre(np.array(edges), fs._T_PER_PANEL)
    return np.exp(t), w


def _grid_arc(u_pi):
    """Log-u grid with [u_pi/2, 3u_pi/2] replaced by a semicircle above u_pi (kind 1) and [u_pi, 3u_pi/2] (kind 2)."""
    lo, hi = np.log(u_pi / 2), np.log(3 * u_pi / 2)
    tl, wl = fs._panel_gauss_legendre(np.array([e for e in fs._T_EDGES if e < lo] + [lo]), fs._T_PER_PANEL)
    tr, wr = fs._panel_gauss_legendre(np.array([hi] + [e for e in fs._T_EDGES if e > hi]), fs._T_PER_PANEL)
    phi, wphi = fs._panel_gauss_legendre(np.linspace(0.0, np.pi, 5), fs._T_PER_PANEL)
    z = u_pi + u_pi / 2 * np.exp(1j * phi)
    x, wx = fs._panel_gauss_legendre(np.array([u_pi, 3 * u_pi / 2]), fs._T_PER_PANEL)
    u = np.concatenate([np.exp(tl), np.exp(tr), z, x])
    w = np.concatenate([wl, wr, -1j * u_pi / 2 * np.exp(1j * phi) * wphi / z, wx / x])
    kind = np.concatenate([np.zeros(len(tl) + len(tr), int), np.ones(len(z), int), 2 * np.ones(len(x), int)])
    return u, w, kind


def build_kernel_table_c(meff_over_H, rho_over_H, c):
    """The FastShapes kernel table for curvature sound speed c != 1 (isocurvature sound speed 1)."""
    lam, c = float(rho_over_H), float(c)
    M2 = float(meff_over_H)**2 - lam**2
    d, v = 1 - c**2, min(c, 1.0)
    u_pi = (c / v - 1) / 2
    A_pi, A_sig = lam**2 * c**2 / d, 2 * lam * c**2.5 / d
    if c < 1:
        u_sigma = (1 - c) / (2 * c)
        u, w = _grid(u_sigma)
        hin, hout = _h_on_nodes(c, M2, lam, c * (1 + 2 * u))
        Om_in = A_pi * (1 + u) * hin
        out = u > u_sigma
        Om_out = np.where(out, A_sig * (1 + u) * hout.real, 0.0)
        i0 = np.argmin(np.where(out, u, np.inf))
        chi = np.arcsinh(Om_in.imag[i0] / Om_out[i0]); cosh_chi = np.cosh(chi)
        omega = (Om_in.real + 1j * cosh_chi * Om_out) / np.sqrt(cosh_chi)
        head_W = 1 / np.sqrt(cosh_chi)
    else:
        u, w, kind = _grid_arc(u_pi)
        ub = (u - u_pi) / c
        hin, hout = _h_on_nodes(c, M2, lam, 1 + 2 * u)
        Om_sig = np.where(kind != 2, A_sig * (1 + ub) * hin, 0.0)
        out = (kind != 1) & (u.real > u_pi)
        Om_pi = np.where(out, A_pi * (1 + ub) * hout.real, 0.0)
        i0 = np.argmin(np.where((kind == 0) & out, u.real, np.inf))
        chi = np.arcsinh(Om_sig.imag[i0] / Om_pi[i0]); cosh_chi = np.cosh(chi)
        omega = (Om_sig + 1j * np.exp(-chi) * Om_pi) / np.sqrt(cosh_chi) * (1 + u) / (1 + u + u_pi)
        u, w = np.append(u, u_pi), np.append(w, 1 / u_pi)
        omega = np.append(omega, 1j * np.exp(-chi) * (1 + u_pi) / np.sqrt(cosh_chi))
        head_W = 0.0
    r_plus = complex(np.sum(w * u * omega / (1 + u)) + head_W)
    return dict(u=u, w=w, omega=omega, omega_V=omega * (1 - u_pi * (1 + u_pi) / (u * (1 + u))), head_W=complex(head_W),
                head_V=0j, r_plus=r_plus, R=float(cosh_chi * abs(r_plus)**2), lam=lam,
                nu_eff=np.sqrt(2.25 - meff_over_H**2 + 0j), chi=float(chi), c=c, v=v)
