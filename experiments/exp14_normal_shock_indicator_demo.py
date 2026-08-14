#!/usr/bin/env python3
"""
exp14_normal_shock_indicator_demo.py

Purpose
-------
Demonstrate how the local rarefaction indicator

    K = max(K_rho, K_T, K_u)

activates inside a shock-like layer even when the global Knudsen number
is small. This experiment is not a DSMC/DVM validation. It is a diagnostic
experiment that connects the article's local K_face theory to a
high-gradient normal-shock setting.

Outputs
-------
results/data/exp14_normal_shock_indicator_profiles.csv
results/data/exp14_normal_shock_indicator_metrics.csv
results/figures/fig37_normal_shock_local_indicator.png
results/figures/fig38_normal_shock_weight_activation.png
results/figures/fig39_normal_shock_global_vs_local_kn.png
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc


OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def wf_log_gaussian(K, K0=0.03, sigma=2.5):
    K = np.asarray(K, dtype=float)
    Ksafe = np.clip(K, 1e-14, 1e14)
    z = np.log(Ksafe / K0) / (np.sqrt(2.0) * sigma)
    return 0.5 * erfc(-z)


def wc_log_gaussian(K, K0=0.03, sigma=2.5):
    return 1.0 - wf_log_gaussian(K, K0=K0, sigma=sigma)


def normal_shock_downstream(M1: float, gamma: float = 1.4):
    """
    Rankine-Hugoniot normal shock relations for ideal gas.

    Upstream nondimensional state:
        rho1 = 1
        T1   = 1
        R    = 1
    """
    rho1 = 1.0
    T1 = 1.0
    R = 1.0
    p1 = rho1 * R * T1
    a1 = math.sqrt(gamma * R * T1)
    u1 = M1 * a1

    rho2_rho1 = ((gamma + 1.0) * M1**2) / ((gamma - 1.0) * M1**2 + 2.0)
    p2_p1 = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1**2 - 1.0)
    T2_T1 = p2_p1 / rho2_rho1

    rho2 = rho1 * rho2_rho1
    p2 = p1 * p2_p1
    T2 = T1 * T2_T1
    u2 = u1 * rho1 / rho2

    return {
        "rho1": rho1,
        "T1": T1,
        "p1": p1,
        "u1": u1,
        "rho2": rho2,
        "T2": T2,
        "p2": p2,
        "u2": u2,
        "a1": a1,
    }


def tanh_profile(x, left, right, delta):
    """
    Smooth monotone transition from left state at x << 0
    to right state at x >> 0.
    """
    S = 0.5 * (1.0 + np.tanh(x / delta))
    return left + (right - left) * S


def compute_local_indicators(x, rho, T, u, Kn_global, gamma=1.4, omega=0.5):
    """
    Compute local gradient-based Knudsen indicators.

    Approximate local mean free path scaling:
        lambda ~ lambda_ref * T^omega / rho

    This is only a diagnostic scaling. It is enough for showing that
    local gradients can make K_local much larger than global Kn.
    """
    dx = x[1] - x[0]

    drho = np.gradient(rho, dx, edge_order=2)
    dT = np.gradient(T, dx, edge_order=2)
    du = np.gradient(u, dx, edge_order=2)

    # Reference lambda = Kn_global because reference length L_ref = 1.
    lam = Kn_global * (T / T[0]) ** omega / np.maximum(rho / rho[0], 1e-14)

    a = np.sqrt(gamma * T)

    K_rho = lam * np.abs(drho) / np.maximum(rho, 1e-14)
    K_T = lam * np.abs(dT) / np.maximum(T, 1e-14)
    K_u = lam * np.abs(du) / np.maximum(np.abs(u) + a, 1e-14)

    K_local = np.maximum.reduce([K_rho, K_T, K_u])

    return lam, K_rho, K_T, K_u, K_local


def layer_width(x, y, threshold):
    mask = y > threshold
    if not np.any(mask):
        return 0.0
    return float(x[mask].max() - x[mask].min())


def main():
    gamma = 1.4

    # Cases are chosen to show:
    # - dependence on Mach number;
    # - local activation can occur even for small global Kn;
    # - fitted parameters activate earlier/broader than default.
    Mach_values = [2.0, 3.0, 5.0]
    Kn_values = [0.001, 0.01, 0.1]

    # x uses reference macroscopic length L_ref = 1.
    x = np.linspace(-0.5, 0.5, 1601)

    profiles = []
    metrics = []

    for M1 in Mach_values:
        states = normal_shock_downstream(M1, gamma=gamma)

        for Kn_global in Kn_values:
            # Shock thickness model in reference length.
            # Use several local mean free paths but not thinner than grid resolution.
            # This is a diagnostic shock-like layer, not a solved shock structure.
            dx = x[1] - x[0]
            delta = max(8.0 * Kn_global, 8.0 * dx)

            rho = tanh_profile(x, states["rho1"], states["rho2"], delta)
            T = tanh_profile(x, states["T1"], states["T2"], delta)
            u = tanh_profile(x, states["u1"], states["u2"], delta)

            lam, K_rho, K_T, K_u, K_local = compute_local_indicators(
                x, rho, T, u, Kn_global, gamma=gamma
            )

            Wf_default = wf_log_gaussian(K_local, K0=0.1, sigma=1.0)
            Wf_fitted = wf_log_gaussian(K_local, K0=0.03, sigma=2.5)

            for i in range(len(x)):
                profiles.append(
                    {
                        "M1": M1,
                        "Kn_global": Kn_global,
                        "x": x[i],
                        "rho": rho[i],
                        "T": T[i],
                        "u": u[i],
                        "lambda": lam[i],
                        "K_rho": K_rho[i],
                        "K_T": K_T[i],
                        "K_u": K_u[i],
                        "K_local": K_local[i],
                        "Wf_default": Wf_default[i],
                        "Wf_fitted": Wf_fitted[i],
                    }
                )

            metrics.append(
                {
                    "M1": M1,
                    "Kn_global": Kn_global,
                    "shock_delta": delta,
                    "max_K_rho": float(np.max(K_rho)),
                    "max_K_T": float(np.max(K_T)),
                    "max_K_u": float(np.max(K_u)),
                    "max_K_local": float(np.max(K_local)),
                    "max_Wf_default": float(np.max(Wf_default)),
                    "max_Wf_fitted": float(np.max(Wf_fitted)),
                    "width_K_gt_0p01": layer_width(x, K_local, 0.01),
                    "width_K_gt_0p03": layer_width(x, K_local, 0.03),
                    "width_K_gt_0p1": layer_width(x, K_local, 0.1),
                    "width_Wf_default_gt_0p5": layer_width(x, Wf_default, 0.5),
                    "width_Wf_fitted_gt_0p5": layer_width(x, Wf_fitted, 0.5),
                }
            )

    df = pd.DataFrame(profiles)
    dm = pd.DataFrame(metrics)

    df.to_csv(OUT_DATA / "exp14_normal_shock_indicator_profiles.csv", index=False)
    dm.to_csv(OUT_DATA / "exp14_normal_shock_indicator_metrics.csv", index=False)

    # Figure 37: local indicators for representative M=3 cases.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for ax, Kn_global in zip(axes, Kn_values):
        sub = df[(df["M1"] == 3.0) & (df["Kn_global"] == Kn_global)]
        ax.plot(sub["x"], sub["K_rho"], label=r"$K_\rho$")
        ax.plot(sub["x"], sub["K_T"], label=r"$K_T$")
        ax.plot(sub["x"], sub["K_u"], label=r"$K_u$")
        ax.plot(sub["x"], sub["K_local"], "k--", label=r"$K=\max$")
        ax.axhline(0.03, linestyle=":", linewidth=1.0, label=r"$K=0.03$" if Kn_global == Kn_values[0] else None)
        ax.axhline(0.1, linestyle="-.", linewidth=1.0, label=r"$K=0.1$" if Kn_global == Kn_values[0] else None)
        ax.set_yscale("log")
        ax.set_ylabel("local K")
        ax.set_title(f"Shock-like layer, M1=3, global Kn={Kn_global:g}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("x / L_ref")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig37_normal_shock_local_indicator.png", dpi=220)
    plt.close(fig)

    # Figure 38: Wf activation default vs fitted for representative M=3 cases.
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for ax, Kn_global in zip(axes, Kn_values):
        sub = df[(df["M1"] == 3.0) & (df["Kn_global"] == Kn_global)]
        ax.plot(sub["x"], sub["Wf_default"], label=r"default $K_0=0.1,\sigma=1$")
        ax.plot(sub["x"], sub["Wf_fitted"], label=r"fitted $K_0=0.03,\sigma=2.5$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(r"$W_f$")
        ax.set_title(f"Ballistic activation, M1=3, global Kn={Kn_global:g}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("x / L_ref")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig38_normal_shock_weight_activation.png", dpi=220)
    plt.close(fig)

    # Figure 39: global Kn versus max local K.
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for M1 in Mach_values:
        sub = dm[dm["M1"] == M1]
        ax.loglog(sub["Kn_global"], sub["max_K_local"], "o-", label=f"M1={M1:g}")
    ax.loglog(Kn_values, Kn_values, "k--", label="global Kn")
    ax.set_xlabel("global Kn")
    ax.set_ylabel("max local K in shock-like layer")
    ax.set_title("Local shock indicator can exceed global Kn")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig39_normal_shock_global_vs_local_kn.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(" ", OUT_DATA / "exp14_normal_shock_indicator_profiles.csv")
    print(" ", OUT_DATA / "exp14_normal_shock_indicator_metrics.csv")
    print(" ", OUT_FIG / "fig37_normal_shock_local_indicator.png")
    print(" ", OUT_FIG / "fig38_normal_shock_weight_activation.png")
    print(" ", OUT_FIG / "fig39_normal_shock_global_vs_local_kn.png")
    print()
    print("Metric summary:")
    print(dm.to_string(index=False))


if __name__ == "__main__":
    main()
