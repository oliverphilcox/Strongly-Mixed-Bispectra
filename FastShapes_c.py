"""Kernel table of FastShapes for unequal sound speeds, c_pi = c != 1, c_sigma = 1.

With s = c(1+2u), u_sigma = (1-c)/(2c), and h_c, h_1 the solutions of the Heun spectral equation analytic at s = c
and s = 1 (h_c(c) = h_1(1) = 1), continued along the real axis from above, the field weights are
    Omega_pi    = delta_+(u) + lam^2 c^2/(1-c^2) (1+u) Theta(u) h_c(s+i0),
    Omega_sigma = 2 lam c^{5/2}/(1-c^2) (1+u) Theta(u-u_sigma) h_1(s+i0),
Im Omega_X = sinh(chi) Omega_Y with (X, Y) = (pi, sigma) for c < 1 and (sigma, pi) for c > 1, and the channel weight is
    omega_+ = (Re Omega_X + i cosh(chi) Omega_Y) / sqrt(cosh chi),    r_+ = int omega_+/(1+u),    R = cosh(chi) |r_+|^2
(mode_functions_unequal_cs.tex, Eqs. 9'-9'''', 22', 11'). The table has FastShapes' layout plus the quadrature weights
'w' (in FastShapes' convention, int f du = sum w u f; the u grid is graded towards u_sigma), 'chi' and 'c', which
FastShapes' leg functions read when present.
"""
import numpy as np
from numpy.polynomial import Polynomial, polynomial as Pn
from scipy.integrate import solve_ivp
import FastShapes as fs

_POLE_CUT = 1e-7   # c > 1: nodes with |u| < _POLE_CUT |u_sigma| around the pole of h_1(s+i0) at u = 0 are dropped


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
    """h_in(s+i0) and h_out(s) at the real nodes s: the solutions analytic at r_in = min(c,1) and r_out = max(c,1),
    the first continued past r_out along a semicircle in the upper half plane; h_out = 0 for s <= r_out."""
    P = _polys(c, M2, lam)
    r_in, r_out = min(c, 1.0), max(c, 1.0)
    a_in, d_in = _taylor(c, M2, lam, r_in); a_out, d_out = _taylor(c, M2, lam, r_out)
    t_in, t_out, rd = 0.2 * d_in, 0.2 * d_out, 0.5 * min(r_out - r_in, 1.0)
    hin, hout = np.zeros(len(s), complex), np.zeros(len(s), complex)
    order = np.argsort(s)
    sel = lambda lo, hi: [i for i in order if lo < s[i] <= hi]

    def march(y, s0, idx, h):
        for i in idx:
            y = _transport(P, y, s0, s[i]); s0 = s[i]; h[i] = y[0]
        return y, s0

    for i in sel(r_in, r_in + t_in):
        hin[i] = _eval(a_in, d_in, s[i] - r_in)[0]
    y, s0 = march(_eval(a_in, d_in, t_in), r_in + t_in, sel(r_in + t_in, r_out - rd), hin)
    y = _transport(P, y, s0, r_out - rd)
    march(y, r_out - rd, sel(r_out - rd, r_out), hin)
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
    """Nodes u and weights w (int f du = sum w u f) of FastShapes' log-u panel grid, graded towards u_sigma (c < 1) or,
    for c > 1, mirrored to u < 0 so that the principal-value pole of h_1(s+i0) at u = 0 cancels pairwise, plus one
    linear panel down to u_sigma."""
    edges = list(fs._T_EDGES)
    if u_sigma < 0:
        t, w = fs._panel_gauss_legendre(np.array(edges), fs._T_PER_PANEL)
        u = np.exp(t)
        keep = u >= _POLE_CUT * abs(u_sigma)
        u, w = u[keep], w[keep]
        u_last = np.exp(max(e for e in edges if np.exp(e) < abs(u_sigma)))
        mirror = u < u_last
        gx, gw = np.polynomial.legendre.leggauss(fs._T_PER_PANEL)
        a, b = u_sigma, -u_last
        ul, wl = 0.5 * (b - a) * gx + 0.5 * (a + b), 0.5 * (b - a) * gw
        return np.concatenate([u, -u[mirror], ul]), np.concatenate([w, -w[mirror], wl / ul])
    ts = np.log(u_sigma)
    if edges[0] < ts < edges[-1]:
        extra = [ts] + [ts + sgn * fs._T_PANEL * 2.0**(-k) for k in range(18) for sgn in (-1, 1)]
        edges = sorted(set(edges + [e for e in extra if edges[0] < e < edges[-1]]))
    t, w = fs._panel_gauss_legendre(np.array(edges), fs._T_PER_PANEL)
    return np.exp(t), w


def build_kernel_table_c(meff_over_H, rho_over_H, c):
    """The FastShapes kernel table for curvature sound speed c != 1 (isocurvature sound speed 1)."""
    lam, c = float(rho_over_H), float(c)
    M2 = float(meff_over_H)**2 - lam**2
    d, u_sigma = 1 - c**2, (1 - c) / (2 * c)
    u, w = _grid(u_sigma)
    hin, hout = _h_on_nodes(c, M2, lam, c * (1 + 2 * u))
    A_pi, A_sig = lam**2 * c**2 / d, 2 * lam * c**2.5 / d
    A_in, A_out = (A_pi, A_sig) if c < 1 else (A_sig, A_pi)
    Om_in = A_in * (1 + u) * hin
    out = u > max(u_sigma, 0.0)
    Om_out = np.where(out, A_out * (1 + u) * hout.real, 0.0)
    sinh_chi = np.median(Om_in.imag[out] / Om_out[out])
    chi = np.arcsinh(sinh_chi); cosh_chi = np.cosh(chi)
    omega = (Om_in.real + 1j * cosh_chi * Om_out) / np.sqrt(cosh_chi)
    if c < 1:
        head_W = 1 / np.sqrt(cosh_chi)
    else:
        eps = _POLE_CUT * abs(u_sigma)
        head_W = (1j * cosh_chi - sinh_chi / np.pi * A_pi * 2 * eps * (np.log(2 * c * eps) - 1)) / np.sqrt(cosh_chi)
    r_plus = complex(np.sum(w * u * omega / (1 + u)) + head_W)
    return dict(u=u, w=w, omega=omega, head_W=complex(head_W), head_V=0j, r_plus=r_plus,
                R=float(cosh_chi * abs(r_plus)**2), lam=lam, nu_eff=np.sqrt(2.25 - meff_over_H**2 + 0j),
                chi=float(chi), c=c)
