"""Kernel table of FastShapes on the real-time contour for curvature sound speed c (c_sigma = 1), heavy or light effective fields.

In u with s = s0 (1+2u), s0 = min(c,1), S = max(c,1), the curvature and isocurvature weights are
    Omega_pi = delta + lam^2 c^2/(1-c^2) (1+u_c) h_c(s),    Omega_sigma = 2 lam c^{5/2}/(1-c^2) (1+u_c) h_1(s),
with s = c(1+2u_c), times (1+u)/(1+u+u_pi) for c > 1, and h_c, h_1 the solutions of the Heun spectral equation analytic at
s = c, 1 (h = 1 there). With (X, Y) = (pi, sigma) for c < 1 and (sigma, pi) for c > 1, Im Omega_X(s+i0) = sinh(chi) Omega_Y,
and the channel sum is taken in the weight,
    rho = [s_c Omega_X(s-i0) + 2 Im r_+ Omega_Y] / sqrt(cosh chi),    s_c = e^chi r_+ + e^-chi r_+^*,    R = cosh(chi) |r_+|^2,
tabulated on the rays u = -it (X) and u_Y - it (Y). The shapes are evaluated on the Schwinger contour [0, iY] u [iY, iY+inf)
with the legs L_K(z) = e^{-iz} Q_K(2iz); on [0, iY] Im L_K is the canonical commutator, Im L_W0 = -(2c/s0) h(z), h the odd
solution of the master equation with h = z + O(z^3).
Extra table keys: 'rho', 'rho_V', 'w' (int f du = sum w u f), 'xi', 'w_N2', 'w_N0', 'n_rt', 'commutator', 'chi', 'c', 'v' = s0.
"""
import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial, polynomial as Pn, chebyshev as Ch
from scipy.integrate import solve_ivp
import FastShapes as fs

# spectral-weight rays t <= e^_T_MAX (the remainder is the asymptotic tail); contour xi = iy with log panels of width _LOG_PANEL from
# e^_Y_T_MIN to _Y_LIN and linear panels of _PANEL light-cone units up to Y, then xi = iY + x linear to _X_LIN and log panels to _X_MAX
_T_MAX, _Y_T_MIN, _LOG_PANEL, _Y_LIN, _PANEL, _X_LIN, _X_MAX = 26.0, -15.0, 1.6, 0.3, 4.0, 0.2, 200.0


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


def _h_ray(c, M2, lam, r, s):
    """The solution analytic at s = r (h(r) = 1) at the nodes s of a ray leaving r, in order."""
    P = _polys(c, M2, lam); a, dist = _taylor(c, M2, lam, r)
    far = abs(s - r) > 0.2 * dist; h = np.zeros(len(s), complex)
    h[~far] = [_eval(a, dist, si - r)[0] for si in s[~far]]
    far = np.flatnonzero(far); s0 = r + 0.2 * dist * (s[far[0]] - r) / abs(s[far[0]] - r)
    y = _eval(a, dist, s0 - r)
    for i in far:
        y = _transport(P, y, s0, s[i]); s0 = s[i]; h[i] = y[0]
    return h


def _commutator(c, s0, meff2, M2, zmax, z0=0.3, N=60, pw=0.25, nc=18):
    """Im L_K(z): h is the odd solution of the master equation (series below z0, DOP853 + Chebyshev above)."""
    A, E, F, G = (1 + c**2) / s0**2, 2 * c**2 / s0**2, c**2 / s0**4, c**2 * (M2 - 2) / s0**2
    a = np.zeros(N); a[0] = 1.0
    for n in range(2, N, 2):
        x = n + 1.0
        a[n] = -((A*(x-2)*(x-3) + E*(x-2) + G) * a[n-2] + (F * a[n-4] if n >= 4 else 0.0)) / (x*(x-1)*(x*x - 3*x + meff2))
    ex = np.arange(N) + 1.0; D = [a, a * ex, a * ex * (ex - 1), a * ex * (ex - 1) * (ex - 2)]
    ser = lambda z, k: (D[k] * np.asarray(z)[..., None]**(ex - k)).sum(-1)
    rhs = lambda z, y: [y[1], y[2], y[3], -(2*z*y[3] + (A*z*z + meff2 - 2)*y[2] + E*z*y[1] + (F*z*z + G)*y[0]) / (z*z)]
    sol = solve_ivp(rhs, (z0, zmax), [ser(z0, k) for k in range(4)], method='DOP853', rtol=1e-13, atol=1e-300, dense_output=True)
    edges = np.append(np.arange(z0, zmax, pw), zmax); mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    xk = np.cos(np.pi * (np.arange(nc) + 0.5) / nc)
    vals = sol.sol((mid[:, None] + half[:, None] * xk).ravel()).reshape(4, len(mid), nc)
    coef = Ch.chebfit(xk, vals.transpose(2, 0, 1).reshape(nc, -1), nc - 1).reshape(nc, 4, len(mid))
    k, q, v = 2 * c / s0, (c / s0)**2, s0

    def commutator(kernel, z):
        H = np.empty((4, len(z))); m = z >= z0
        H[:, ~m] = [ser(z[~m], j) for j in range(4)]
        p = np.minimum(np.searchsorted(edges, z[m], side='right') - 1, len(mid) - 1)
        H[:, m] = np.einsum('ik,kji->ji', Ch.chebvander((z[m] - mid[p]) / half[p], nc - 1), coef[:, :, p])
        if kernel == 'W2':
            return k * H[2]
        if kernel == 'V':
            return k / 4 * (H[2] + q * H[0])
        if kernel == 'U':
            return k / 4 * (2 * (H[2] + q * H[0]) + z * (H[3] + q * H[1]))
        return -k * (H[0] - z * H[1]) / v
    return commutator


def build_kernel_table_c(meff_over_H, rho_over_H, c, Y=10.0):
    """The FastShapes kernel table for curvature sound speed c (isocurvature sound speed 1) on the real-time contour, which also
    converges for light effective fields; at c = 1 the weights are FastShapes' closed-form omega_a on the ray u = -it."""
    lam, c = float(rho_over_H), float(c)
    M2 = float(meff_over_H)**2 - lam**2; nu = np.sqrt(2.25 - meff_over_H**2 + 0j)
    edges = fs._T_EDGES[fs._T_EDGES <= _T_MAX]; t, w = fs._panel_gauss_legendre(edges, fs._T_PER_PANEL); t, te = np.exp(t), np.exp(edges[-1])
    if c == 1:
        s0, u_pi, chi, p, nu_mp = 1.0, 0.0, np.pi * lam / 2, 1j * lam / 2, mp.mpc(nu.real, nu.imag)
        u = -1j * t; om = [np.array([complex(fs._omega_a(mp.mpc(x.real, x.imag), a, lam, nu_mp)) for x in u]) for a in (1, -1)]
        r_plus = complex(fs._r_a(1, lam, nu_mp)); R = float(fs._power_spectrum_ratio(lam, nu_mp))
        rho = np.exp(chi) * r_plus * om[1] + np.exp(-chi) * np.conj(r_plus) * om[0]
        umin = -1j * np.exp(edges[0]); cm, cp = np.exp(chi) * r_plus / complex(mp.gamma(-p)), np.exp(-chi) * np.conj(r_plus) / complex(mp.gamma(p))
        head_W = cm * umin**(-p) / (-p) + cp * umin**p / p; head_V = cm * umin**(1 - p) / (1 - p) + cp * umin**(1 + p) / (1 + p)
    else:
        s0, S = min(c, 1.0), max(c, 1.0)
        u_pi, u_Y = (c / s0 - 1) / 2, (S / s0 - 1) / 2
        A_pi, A_sig = lam**2 * c**2 / (1 - c**2), 2 * lam * c**2.5 / (1 - c**2)
        A_X, A_Y = (A_pi, A_sig) if c < 1 else (A_sig, A_pi)
        # chi: h_X continued below the upper cone to s* = S + dist/5, where h_Y is its Taylor series
        aY, dY = _taylor(c, M2, lam, S); s_star = S + 0.2 * dY
        hX = _h_ray(c, M2, lam, s0, np.array([s0 - 0.5j * dY, S - 0.5j * dY, s_star]))[-1]
        chi = np.arcsinh(-A_X * hX.imag / (A_Y * _eval(aY, dY, 0.2 * dY)[0].real)); cosh_chi = np.cosh(chi)
        # the rays u = -it and u_Y - it, t = e^{t-nodes of FastShapes} <= e^_T_MAX
        uX, uY = -1j * t, u_Y - 1j * t
        ub = lambda u: (u - u_pi) / c if c > 1 else u
        f = lambda u: (1 + u) / (1 + u + u_pi) if c > 1 else 1.0
        Om_X = A_X * (1 + ub(uX)) * f(uX) * _h_ray(c, M2, lam, s0, s0 * (1 + 2 * uX))
        Om_Y = A_Y * (1 + ub(uY)) * f(uY) * _h_ray(c, M2, lam, S, s0 * (1 + 2 * uY))
        u, w = np.concatenate([uX, uY]), np.concatenate([w, w * (uY - u_Y) / uY])
        r_plus = np.sum(w * u * np.concatenate([Om_X, 1j * np.exp(chi) * Om_Y]) / (1 + u))
        p = nu - 2.5; tail = lambda Om, ul, ue: Om[-1] / (1 + ul) * ((1 + 2 * ue) / (1 + 2 * ul))**p * (1 + 2 * ue) / (-2 * (p + 1))
        r_plus += tail(Om_X, uX[-1], -1j * te) + 1j * np.exp(chi) * tail(Om_Y, uY[-1], u_Y - 1j * te)     # the rays beyond the last panel
        d_X, d_Y = (1.0, 0.0) if c < 1 else (0.0, 1 + u_pi)                       # the curvature delta, on X or on Y
        r_plus = (r_plus + (d_X + 1j * np.exp(chi) * d_Y) / (1 + u_pi)) / np.sqrt(cosh_chi)
        s_c = np.exp(chi) * r_plus + np.exp(-chi) * np.conj(r_plus)
        rho = np.concatenate([s_c * Om_X, 2 * r_plus.imag * Om_Y]) / np.sqrt(cosh_chi)
        if c > 1:                                                                  # delta at u_pi as a node: w u = 1
            u, w, rho = np.append(u, u_pi), np.append(w, 1 / u_pi), np.append(rho, 2 * r_plus.imag * d_Y / np.sqrt(cosh_chi))
        head_W, head_V, R = s_c * d_X / np.sqrt(cosh_chi), 0j, float(cosh_chi * abs(r_plus)**2)
    gl = lambda e: fs._panel_gauss_legendre(np.asarray(e, float), fs._XI_T_PER_PANEL)
    ea = np.append(np.arange(_Y_T_MIN, np.log(_Y_LIN), _LOG_PANEL), np.log(_Y_LIN)); ya, wa = gl(ea); ya = np.exp(ya)
    yb, wb = gl(np.linspace(_Y_LIN, Y, int(np.ceil((Y - _Y_LIN) * max(c, 1 / c) / _PANEL)) + 1))
    xa, wxa = gl([0.0, _X_LIN]); xb, wxb = gl(np.append(np.arange(np.log(_X_LIN), np.log(_X_MAX), _LOG_PANEL), np.log(_X_MAX))); xb = np.exp(xb)
    xi = np.concatenate([1j * ya, 1j * yb, 1j * Y + xa, 1j * Y + xb])
    dxi = np.concatenate([1j * ya * wa, 1j * wb, wxa, xb * wxb])
    w2 = dxi * xi**2; w2[:fs._XI_T_PER_PANEL] /= -np.expm1(-(ea[1] - ea[0]) * (3 - 2 * nu))     # y < e^_Y_T_MIN: integrand ~ y^(2-2nu)
    return dict(u=u, w=w, rho=rho, rho_V=rho * (1 - u_pi * (1 + u_pi) / (u * (1 + u))), head_W=head_W, head_V=head_V,
                r_plus=complex(r_plus), R=R, lam=lam, nu_eff=nu, chi=float(chi), c=c, v=s0,
                xi=xi, w_N2=w2, w_N0=dxi, n_rt=len(ya) + len(yb),
                commutator=_commutator(c, s0, float(meff_over_H)**2, M2, Y / 2 + 0.5))
