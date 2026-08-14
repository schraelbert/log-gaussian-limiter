import numpy as np


def ideal_gas_pressure(rho, r=1.0, t=1.0):
    return rho * r * t


def sound_speed(t, gamma=1.4, r=1.0):
    return np.sqrt(gamma * r * t)


def mean_free_path_simple(kn_global=0.01, l_ref=1.0):
    """Simple nondimensional scaling lambda = Kn_global * L_ref."""
    return kn_global * l_ref


def rankine_hugoniot_downstream(m1, gamma=1.4, rho1=1.0, t1=1.0, r=1.0):
    """Normal shock downstream state for an ideal gas."""
    p1 = rho1 * r * t1
    a1 = np.sqrt(gamma * r * t1)
    u1 = m1 * a1
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (m1**2 - 1.0)
    rho2_rho1 = ((gamma + 1.0) * m1**2) / ((gamma - 1.0) * m1**2 + 2.0)
    t2_t1 = p2_p1 / rho2_rho1
    rho2 = rho1 * rho2_rho1
    t2 = t1 * t2_t1
    u2 = rho1 * u1 / rho2
    p2 = p1 * p2_p1
    return dict(rho1=rho1, t1=t1, p1=p1, u1=u1,
                rho2=rho2, t2=t2, p2=p2, u2=u2,
                m1=m1, gamma=gamma, r=r)
