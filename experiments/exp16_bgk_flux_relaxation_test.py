#!/usr/bin/env python3
"""
exp16_bgk_flux_relaxation_test.py

Purpose
-------
Validate the BGK-inspired kinetic flux relaxation coefficient

    alpha = tau/dt * (1 - exp(-dt/tau))

against a direct discrete-velocity time-integrated BGK relaxation model.

This is not a full spatial DVM solver. It is an interface-flux diagnostic:
for a left/right Maxwellian Riemann-like state, the free-transport
upwind distribution relaxes toward an equilibrium Maxwellian over a
time interval dt. The direct time-averaged flux is compared with

    F_model = alpha F_FM + (1-alpha) F_EQ.

Outputs
-------
results/data/exp16_bgk_flux_relaxation.csv
results/data/exp16_bgk_flux_relaxation_summary.csv
results/figures/fig42_bgk_flux_relaxation_alpha.png
results/figures/fig43_bgk_flux_relaxation_errors.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def maxwellian_1d(v, rho, u, T):
    return rho / np.sqrt(2.0 * np.pi * T) * np.exp(-0.5 * (v - u) ** 2 / T)


def moments_from_distribution(v, f):
    dv = v[1] - v[0]
    rho = np.sum(f) * dv
    mom = np.sum(v * f) * dv
    E = np.sum(0.5 * v**2 * f) * dv
    u = mom / max(rho, 1e-300)
    T = max(2.0 * (E / max(rho, 1e-300) - 0.5 * u**2), 1e-12)
    return rho, u, T, mom, E


def flux_from_distribution(v, f):
    dv = v[1] - v[0]
    return np.array([
        np.sum(v * f) * dv,
        np.sum(v**2 * f) * dv,
        np.sum(0.5 * v**3 * f) * dv,
    ])


def half_range_free_molecular_flux(v, fL, fR):
    f_up = np.where(v > 0.0, fL, fR)
    return flux_from_distribution(v, f_up), f_up


def equilibrium_flux_from_interface_state(v, f_up):
    rho, u, T, _, _ = moments_from_distribution(v, f_up)
    f_eq = maxwellian_1d(v, rho, u, T)
    return flux_from_distribution(v, f_eq), f_eq


def alpha(dt_over_tau):
    r = np.asarray(dt_over_tau, dtype=float)
    # alpha = tau/dt * (1-exp(-dt/tau)) = (1-exp(-r))/r
    return np.where(r > 1e-14, (1.0 - np.exp(-r)) / r, 1.0)


def direct_time_averaged_bgk_flux(v, f_up, f_eq, dt_over_tau):
    """
    Model distribution:
        f(t) = f_eq + exp(-t/tau) * (f_up - f_eq)

    Time average over [0,dt]:
        <f> = f_eq + alpha * (f_up - f_eq)

    This is analytically equivalent to the model flux. Here we compute it
    explicitly on the velocity grid as a consistency diagnostic.
    """
    a = alpha(dt_over_tau)
    f_avg = f_eq + a * (f_up - f_eq)
    return flux_from_distribution(v, f_avg)


def main():
    v = np.linspace(-9.0, 9.0, 6001)

    cases = [
        {
            "case": "mild_contact",
            "rhoL": 1.0,
            "uL": 0.3,
            "TL": 1.0,
            "rhoR": 0.7,
            "uR": -0.1,
            "TR": 1.2,
        },
        {
            "case": "strong_temp_jump",
            "rhoL": 1.0,
            "uL": 0.2,
            "TL": 0.8,
            "rhoR": 0.4,
            "uR": -0.2,
            "TR": 2.0,
        },
        {
            "case": "counter_streaming",
            "rhoL": 1.0,
            "uL": 1.0,
            "TL": 0.7,
            "rhoR": 1.0,
            "uR": -1.0,
            "TR": 0.7,
        },
    ]

    dt_over_tau_values = np.logspace(-3, 3, 121)

    rows = []

    for c in cases:
        fL = maxwellian_1d(v, c["rhoL"], c["uL"], c["TL"])
        fR = maxwellian_1d(v, c["rhoR"], c["uR"], c["TR"])

        F_FM, f_up = half_range_free_molecular_flux(v, fL, fR)
        F_EQ, f_eq = equilibrium_flux_from_interface_state(v, f_up)

        for r in dt_over_tau_values:
            a = float(alpha(r))
            F_model = a * F_FM + (1.0 - a) * F_EQ
            F_direct = direct_time_averaged_bgk_flux(v, f_up, f_eq, r)

            denom = np.linalg.norm(F_direct) + 1e-14
            rel = np.linalg.norm(F_model - F_direct) / denom

            rows.append(
                {
                    "case": c["case"],
                    "dt_over_tau": r,
                    "alpha": a,
                    "rel_error_total_flux": rel,
                    "mass_flux_direct": F_direct[0],
                    "momentum_flux_direct": F_direct[1],
                    "energy_flux_direct": F_direct[2],
                    "mass_flux_model": F_model[0],
                    "momentum_flux_model": F_model[1],
                    "energy_flux_model": F_model[2],
                    "mass_flux_FM": F_FM[0],
                    "momentum_flux_FM": F_FM[1],
                    "energy_flux_FM": F_FM[2],
                    "mass_flux_EQ": F_EQ[0],
                    "momentum_flux_EQ": F_EQ[1],
                    "energy_flux_EQ": F_EQ[2],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "exp16_bgk_flux_relaxation.csv", index=False)

    summary = (
        df.groupby("case", as_index=False)
        .agg(
            max_rel_error=("rel_error_total_flux", "max"),
            mean_rel_error=("rel_error_total_flux", "mean"),
            alpha_min=("alpha", "min"),
            alpha_max=("alpha", "max"),
        )
    )
    summary.to_csv(OUT_DATA / "exp16_bgk_flux_relaxation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.loglog(dt_over_tau_values, alpha(dt_over_tau_values))
    ax.set_xlabel(r"$\Delta t/\tau$")
    ax.set_ylabel(r"$\alpha=(1-e^{-\Delta t/\tau})/(\Delta t/\tau)$")
    ax.set_title("BGK relaxation coefficient")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig42_bgk_flux_relaxation_alpha.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for case, sub in df.groupby("case"):
        ax.loglog(sub["dt_over_tau"], np.maximum(sub["rel_error_total_flux"], 1e-16), label=case)
    ax.set_xlabel(r"$\Delta t/\tau$")
    ax.set_ylabel("relative flux error")
    ax.set_title("Model flux versus direct time-averaged BGK flux")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig43_bgk_flux_relaxation_errors.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(" ", OUT_DATA / "exp16_bgk_flux_relaxation.csv")
    print(" ", OUT_DATA / "exp16_bgk_flux_relaxation_summary.csv")
    print(" ", OUT_FIG / "fig42_bgk_flux_relaxation_alpha.png")
    print(" ", OUT_FIG / "fig43_bgk_flux_relaxation_errors.png")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
