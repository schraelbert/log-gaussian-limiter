import numpy as np
from scipy.special import erfc


def _safe_k(k):
    return np.maximum(np.asarray(k, dtype=float), 1.0e-300)


def wc(k, k0=0.1, sigma=1.0):
    """Continuum/collisional log-Gaussian weight."""
    k = _safe_k(k)
    s = np.log(k / k0)
    return 0.5 * erfc(s / (np.sqrt(2.0) * sigma))


def wf(k, k0=0.1, sigma=1.0):
    """Ballistic/free-transport log-Gaussian weight."""
    return 1.0 - wc(k, k0=k0, sigma=sigma)


def weights(k, k0=0.1, sigma=1.0):
    c = wc(k, k0=k0, sigma=sigma)
    return c, 1.0 - c


def algebraic_wf(k, k0=0.1, m=2.0):
    """Baseline algebraic switch for ablation comparison."""
    k = _safe_k(k)
    return k**m / (k**m + k0**m)


def hard_wf(k, k0=0.1):
    """Baseline hard switch for ablation comparison."""
    k = np.asarray(k, dtype=float)
    return (k >= k0).astype(float)
