#!/usr/bin/env python3
"""
exp19_dvm_bgk_normal_shock.py

Reduced DVM/BGK normal-shock diagnostic.

This is a 1D-in-space, 1D-in-velocity BGK solver:

    df/dt + v df/dx = (M[f] - f)/tau

It is used as a kinetic shock-layer diagnostic for the log-Gaussian
local rarefaction indicator. It is not DSMC, not a full hypersonic
2D blunt-body validation, and not a full UGKS/AP proof.

Outputs
-------
results/data/exp19_dvm_bgk_normal_shock_profiles.csv
results/data/exp19_dvm_bgk_normal_shock_metrics.csv
results/figures/fig48_exp19_normal_shock_profiles.png
results/figures/fig49_exp19_normal_shock_indicator_weights.png
results/figures/fig50_exp19_normal_shock_convergence.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt

OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def maxwellian(v, rho, u, T):
    T = max(float(T), 1e-12)
    return rho / np.sqrt(2.0 * np.pi * T) * np.exp(-0.5 * (v - u) ** 2 / T)


def moments(v, f):
    dv = v[1] - v[0]
    rho = np.sum(f, axis=1) * dv
    mom = np.sum(f * v[None, :], axis=1) * dv
    u = mom / np.maximum(rho, 1e-300)
    e = np.sum(0.5 * (v[None, :] ** 2) * f, axis=1) * dv
    T = 2.0 * (e / np.maximum(rho, 1e-300) - 0.5 * u**2)
    T = np.maximum(T, 1e-10)
    return rho, u, T


def make_M(v, rho, u, T):
    M = np.empty((len(rho), len(v)))
    for i in range(len(rho)):
        M[i, :] = maxwellian(v, rho[i], u[i], T[i])
    return M


def normal_shock_states(M1, gamma=3.0, rho1=1.0, T1=1.0):
    a1 = np.sqrt(gamma * T1)
    u1 = M1 * a1
    r = ((gamma + 1.0) * M1**2) / ((gamma - 1.0) * M1**2 + 2.0)
    p2p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M1**2 - 1.0)
    rho2 = rho1 * r
    T2 = T1 * p2p1 / r
    u2 = u1 / r
    return rho1, u1, T1, rho2, u2, T2


def wf_loggauss(K, K0, sigma):
    K = np.maximum(np.asarray(K), 1e-300)
    z = np.log(K / K0) / (sqrt(2.0) * sigma)
    # Wf = 0.5 * erfc(-z) = 0.5 * (1 + erf(z))
    return 0.5 * (1.0 + np.vectorize(erf)(z))


def run_case(Mach=3.0, Kn=0.05, nx=241, nv=301, nsteps=100000, tol=1e-6):
    gamma = 3.0
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]

    rho1, u1, T1, rho2, u2, T2 = normal_shock_states(Mach, gamma=gamma)

    vmax = max(abs(u1), abs(u2)) + 8.0 * np.sqrt(max(T1, T2))
    v = np.linspace(-vmax, vmax, nv)

    fL = maxwellian(v, rho1, u1, T1)
    fR = maxwellian(v, rho2, u2, T2)

    width0 = max(6.0 * Kn, 4.0 * dx)
    s = 0.5 * (1.0 + np.tanh(x / width0))
    rho0 = rho1 * (1.0 - s) + rho2 * s
    u0 = u1 * (1.0 - s) + u2 * s
    T0 = T1 * (1.0 - s) + T2 * s
    f = make_M(v, rho0, u0, T0)

    tau = Kn
    dt_adv = 0.45 * dx / np.max(np.abs(v))
    dt_col = 0.25 * tau
    dt = min(dt_adv, dt_col)

    pos = v > 0.0
    neg = v < 0.0
    history = []

    for it in range(1, nsteps + 1):
        rho, u, T = moments(v, f)
        Mloc = make_M(v, rho, u, T)

        dfdx = np.zeros_like(f)

        # v > 0: backward difference
        dfdx[1:, pos] = (f[1:, pos] - f[:-1, pos]) / dx
        dfdx[0, pos] = (f[0, pos] - fL[pos]) / dx

        # v < 0: forward difference
        dfdx[:-1, neg] = (f[1:, neg] - f[:-1, neg]) / dx
        dfdx[-1, neg] = (fR[neg] - f[-1, neg]) / dx

        rhs = -v[None, :] * dfdx + (Mloc - f) / tau
        f_new = f + dt * rhs
        f_new = np.maximum(f_new, 1e-300)

        # Inflow kinetic boundary conditions
        f_new[0, pos] = fL[pos]
        f_new[-1, neg] = fR[neg]

        if it % 200 == 0:
            rel = np.linalg.norm(f_new - f) / (np.linalg.norm(f) + 1e-300)
            history.append({"Mach": Mach, "Kn": Kn, "step": it, "residual": rel})
            if rel < tol and it > 2000:
                f = f_new
                break

        f = f_new

    rho, u, T = moments(v, f)

    drho = np.gradient(rho, x)
    dT = np.gradient(T, x)
    du = np.gradient(u, x)

    # Simple local mean-free-path proxy. This preserves the intended scaling:
    # denser downstream gas has smaller local molecular length.
    lam = Kn * rho1 / np.maximum(rho, 1e-300)

    a = np.sqrt(gamma * T)
    K_rho = lam * np.abs(drho) / np.maximum(rho, 1e-300)
    K_T = lam * np.abs(dT) / np.maximum(T, 1e-300)
    K_u = lam * np.abs(du) / (np.abs(u) + a + 1e-300)
    K_local = np.maximum.reduce([K_rho, K_T, K_u])

    Wf_default = wf_loggauss(K_local, 0.1, 1.0)
    Wf_fitted = wf_loggauss(K_local, 0.03, 2.5)

    max_drho = np.max(np.abs(drho))
    inv_density_thickness = max_drho / max(abs(rho2 - rho1), 1e-300)
    shock_thickness = 1.0 / max(inv_density_thickness, 1e-300)

    prof = pd.DataFrame(
        {
            "Mach": Mach,
            "Kn": Kn,
            "x": x,
            "rho": rho,
            "u": u,
            "T": T,
            "K_rho": K_rho,
            "K_T": K_T,
            "K_u": K_u,
            "K_local": K_local,
            "Wf_default": Wf_default,
            "Wf_fitted": Wf_fitted,
        }
    )

    met = {
        "Mach": Mach,
        "Kn": Kn,
        "nsteps_used": it,
        "dt": dt,
        "rho1": rho1,
        "u1": u1,
        "T1": T1,
        "rho2": rho2,
        "u2": u2,
        "T2": T2,
        "max_K_local": float(np.max(K_local)),
        "max_Wf_default": float(np.max(Wf_default)),
        "max_Wf_fitted": float(np.max(Wf_fitted)),
        "shock_thickness": float(shock_thickness),
        "inverse_density_thickness": float(inv_density_thickness),
        
"final_residual": history[-1]["residual"] if history else np.nan,
"residual_tol": tol,
"converged": bool(history and history[-1]["residual"] < tol),

    }

    return prof, met, pd.DataFrame(history)


def main():
    cases = [
        (2.0, 0.03),
        (3.0, 0.03),
        (3.0, 0.10),
        (5.0, 0.03),
    ]

    profiles = []
    metrics = []
    histories = []

    for Mach, Kn in cases:
        print(f"running Mach={Mach:g}, Kn={Kn:g}")
        prof, met, hist = run_case(Mach=Mach, Kn=Kn)
        profiles.append(prof)
        metrics.append(met)
        histories.append(hist)
        print(
            f"  steps={met['nsteps_used']} residual={met['final_residual']:.3e} "
            f"maxK={met['max_K_local']:.3e} "
            f"Wf_def={met['max_Wf_default']:.3f} "
            f"Wf_fit={met['max_Wf_fitted']:.3f}"
        )

    dfp = pd.concat(profiles, ignore_index=True)
    dfm = pd.DataFrame(metrics)
    dfh = pd.concat(histories, ignore_index=True)

    dfp.to_csv(OUT_DATA / "exp19_dvm_bgk_normal_shock_profiles.csv", index=False)
    dfm.to_csv(OUT_DATA / "exp19_dvm_bgk_normal_shock_metrics.csv", index=False)
    dfh.to_csv(OUT_DATA / "exp19_dvm_bgk_normal_shock_convergence.csv", index=False)

    # Figure 48: profiles
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for (Mach, Kn), sub in dfp.groupby(["Mach", "Kn"]):
        label = f"M={Mach:g}, Kn={Kn:g}"
        axes[0].plot(sub["x"], sub["rho"], label=label)
        axes[1].plot(sub["x"], sub["u"], label=label)
        axes[2].plot(sub["x"], sub["T"], label=label)
    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$u$")
    axes[2].set_ylabel(r"$T$")
    axes[2].set_xlabel("x")
    axes[0].set_title("Reduced DVM/BGK normal-shock profiles")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig48_exp19_normal_shock_profiles.png", dpi=220)
    plt.close(fig)

    # Figure 49: K and weights
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
    for (Mach, Kn), sub in dfp.groupby(["Mach", "Kn"]):
        label = f"M={Mach:g}, Kn={Kn:g}"
        axes[0].semilogy(sub["x"], np.maximum(sub["K_local"], 1e-12), label=label)
        axes[1].plot(sub["x"], sub["Wf_default"], label=label)
        axes[2].plot(sub["x"], sub["Wf_fitted"], label=label)
    axes[0].set_ylabel(r"$K_{\rm local}$")
    axes[1].set_ylabel(r"$W_f$ default")
    axes[2].set_ylabel(r"$W_f$ fitted")
    axes[2].set_xlabel("x")
    axes[0].set_title("Local rarefaction indicator and kinetic activation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig49_exp19_normal_shock_indicator_weights.png", dpi=220)
    plt.close(fig)

    # Figure 50: convergence
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for (Mach, Kn), sub in dfh.groupby(["Mach", "Kn"]):
        ax.semilogy(sub["step"], sub["residual"], label=f"M={Mach:g}, Kn={Kn:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative update residual")
    ax.set_title("Reduced BGK shock relaxation convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig50_exp19_normal_shock_convergence.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(" ", OUT_DATA / "exp19_dvm_bgk_normal_shock_profiles.csv")
    print(" ", OUT_DATA / "exp19_dvm_bgk_normal_shock_metrics.csv")
    print(" ", OUT_DATA / "exp19_dvm_bgk_normal_shock_convergence.csv")
    print(" ", OUT_FIG / "fig48_exp19_normal_shock_profiles.png")
    print(" ", OUT_FIG / "fig49_exp19_normal_shock_indicator_weights.png")
    print(" ", OUT_FIG / "fig50_exp19_normal_shock_convergence.png")
    print()
    print(dfm.to_string(index=False))


if __name__ == "__main__":
    main()
