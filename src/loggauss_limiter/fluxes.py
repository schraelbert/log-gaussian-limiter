import numpy as np
from scipy.special import erfc
from .weights import wc


def euler_flux_2d(state, n=(1.0, 0.0), gamma=1.4, r=1.0):
    """Euler flux in normal direction for state=(rho, ux, uy, T)."""
    rho, ux, uy, T = state
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.array([ux, uy], dtype=float)
    un = float(np.dot(u, n))
    p = rho * r * T
    e = rho * (r * T / (gamma - 1.0)) + 0.5 * rho * np.dot(u, u)
    mom_flux = rho * un * u + p * n
    e_flux = (e + p) * un
    return np.array([rho * un, mom_flux[0], mom_flux[1], e_flux], dtype=float)


def half_range_moments(rho, un, theta, sign=+1):
    """Half-range moments M0..M3 for xi_n > 0 or xi_n < 0."""
    a = un / np.sqrt(2.0 * theta)
    if sign > 0:
        A = 0.5 * erfc(-a)
        B = np.sqrt(theta / (2.0 * np.pi)) * np.exp(-a*a)
        M0 = rho * A
        M1 = rho * (un * A + B)
        M2 = rho * ((un**2 + theta) * A + un * B)
        M3 = rho * ((un**3 + 3.0 * un * theta) * A + (un**2 + 2.0 * theta) * B)
    else:
        A = 0.5 * erfc(a)
        B = np.sqrt(theta / (2.0 * np.pi)) * np.exp(-a*a)
        M0 = rho * A
        M1 = rho * (un * A - B)
        M2 = rho * ((un**2 + theta) * A - un * B)
        M3 = rho * ((un**3 + 3.0 * un * theta) * A - (un**2 + 2.0 * theta) * B)
    return M0, M1, M2, M3


def half_range_flux_local_2d(state, sign=+1, gamma=1.4, r=1.0):
    """Half-range flux in local n-t coordinates. state=(rho, un, ut, T)."""
    rho, un, ut, T = state
    theta = r * T
    b = 2.0 / (gamma - 1.0) - 2.0
    M0, M1, M2, M3 = half_range_moments(rho, un, theta, sign=sign)
    F_rho = M1
    F_mom_n = M2
    F_mom_t = ut * M1
    F_E = 0.5 * M3 + 0.5 * (ut**2 + (1.0 + b) * theta) * M1
    return np.array([F_rho, F_mom_n, F_mom_t, F_E], dtype=float)


def _normal_tangent(n):
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    t = np.array([-n[1], n[0]], dtype=float)
    return n, t


def to_local_state(state, n=(1.0, 0.0)):
    rho, ux, uy, T = state
    n, t = _normal_tangent(n)
    u = np.array([ux, uy], dtype=float)
    return np.array([rho, np.dot(u, n), np.dot(u, t), T], dtype=float)


def local_flux_to_global(Floc, n=(1.0, 0.0)):
    n, t = _normal_tangent(n)
    F_rho, F_mom_n, F_mom_t, F_E = Floc
    mom = F_mom_n * n + F_mom_t * t
    return np.array([F_rho, mom[0], mom[1], F_E], dtype=float)


def half_range_flux_2d(left, right, n=(1.0, 0.0), gamma=1.4, r=1.0):
    """Free-molecular upwind flux using half-range Maxwellian moments."""
    Lloc = to_local_state(left, n)
    Rloc = to_local_state(right, n)
    Fp = half_range_flux_local_2d(Lloc, sign=+1, gamma=gamma, r=r)
    Fm = half_range_flux_local_2d(Rloc, sign=-1, gamma=gamma, r=r)
    return local_flux_to_global(Fp + Fm, n)


def hybrid_flux_2d(left, right, k_face, tau, dt, n=(1.0, 0.0),
                   k0=0.1, sigma=1.0, gamma=1.4, r=1.0):
    """Hybrid log-Gaussian flux. Continuum branch uses Euler flux in this demo.

    In a full finite-volume code, replace F_NS by reconstructed NSF flux with
    viscous stress and heat flux. This function is intended as a minimal
    executable diagnostic for the paper framework.
    """
    c = wc(k_face, k0=k0, sigma=sigma)
    f = 1.0 - c
    alpha = (tau / dt) * (1.0 - np.exp(-dt / tau)) if tau > 0 else 0.0
    avg = 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))
    F_eq = euler_flux_2d(avg, n=n, gamma=gamma, r=r)
    F_fm = half_range_flux_2d(left, right, n=n, gamma=gamma, r=r)
    F_k = alpha * F_fm + (1.0 - alpha) * F_eq
    F_ns = F_eq
    return c * F_ns + f * F_k
