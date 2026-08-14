#!/usr/bin/env python3
"""
exp20_2d_blunt_body_indicator.py

2D blunt-body bow-shock local rarefaction diagnostic.

This is not a full 2D hypersonic solver. It constructs a smooth
bow-shock/blunt-body diagnostic field and evaluates the local rarefaction
indicator and log-Gaussian kinetic activation weights.

Purpose
-------
Provide a 2D shock-layer geometry diagnostic for the journal version:
curved bow shock, stagnation region, and local-gradient activation.

Outputs
-------
results/data/exp20_2d_blunt_body_metrics.csv
results/data/exp20_2d_blunt_body_fields.csv
results/figures/fig51_exp20_2d_blunt_body_fields.png
results/figures/fig52_exp20_2d_blunt_body_weights.png
results/tables/table_exp20_2d_blunt_body_metrics.tex
"""

from __future__ import annotations

from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_TAB = Path("results/tables")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def normal_shock_states(M1, gamma=1.4, rho1=1.0, T1=1.0):
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
    return 0.5 * (1.0 + np.vectorize(erf)(z))


def make_case(Mach=5.0, Kn=0.03, nx=360, ny=260):
    gamma = 1.4

    x = np.linspace(-3.0, 2.0, nx)
    y = np.linspace(-1.8, 1.8, ny)
    X, Y = np.meshgrid(x, y)

    R = 1.0
    body = X**2 + Y**2 <= R**2

    rho1, u1, T1, rho2, u2, T2 = normal_shock_states(Mach, gamma=gamma)

    # Approximate bow shock in front of the circular body.
    # Flow is from left to right. Shock is upstream of the nose near x=-R.
    standoff = 0.35 + 0.8 * Kn + 0.02 * Mach
    curvature = 0.33
    xshock = -R - standoff + curvature * Y**2

    # Shock thickness grows with Kn. Keep a minimum for grid smoothness.
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    delta = max(8.0 * Kn, 2.5 * dx)

    S = 0.5 * (1.0 + np.tanh((X - xshock) / delta))

    # Upstream -> post-shock interpolation.
    rho = rho1 * (1.0 - S) + rho2 * S
    T = T1 * (1.0 - S) + T2 * S

    # Velocity: slowed normal flow after shock, with a crude stagnation
    # reduction near the body nose.
    ux_base = u1 * (1.0 - S) + u2 * S
    r = np.sqrt(X**2 + Y**2)
    nose_factor = 1.0 - 0.75 * np.exp(-((X + R) ** 2 + Y**2) / 0.25)
    ux = ux_base * np.maximum(nose_factor, 0.05)
    uy = -0.18 * ux_base * S * Y * np.exp(-((X + R + 0.2) ** 2 + Y**2) / 1.6)

    # Mask the solid body.
    rho = np.where(body, np.nan, rho)
    T = np.where(body, np.nan, T)
    ux = np.where(body, np.nan, ux)
    uy = np.where(body, np.nan, uy)

    # Gradients. Replace NaNs by nearest harmless values for gradient
    # calculation, then mask the body again.
    def fill_nan(A):
        B = A.copy()
        m = np.isfinite(B)
        B[~m] = np.nanmean(B)
        return B

    rho_f = fill_nan(rho)
    T_f = fill_nan(T)
    ux_f = fill_nan(ux)
    uy_f = fill_nan(uy)

    drho_dy, drho_dx = np.gradient(rho_f, dy, dx)
    dT_dy, dT_dx = np.gradient(T_f, dy, dx)
    dux_dy, dux_dx = np.gradient(ux_f, dy, dx)
    duy_dy, duy_dx = np.gradient(uy_f, dy, dx)

    grad_rho = np.sqrt(drho_dx**2 + drho_dy**2)
    grad_T = np.sqrt(dT_dx**2 + dT_dy**2)
    grad_u = np.sqrt(dux_dx**2 + dux_dy**2 + duy_dx**2 + duy_dy**2)

    a = np.sqrt(gamma * np.maximum(T_f, 1e-12))
    speed = np.sqrt(ux_f**2 + uy_f**2)

    lam = Kn * rho1 / np.maximum(rho_f, 1e-300)

    K_rho = lam * grad_rho / np.maximum(rho_f, 1e-300)
    K_T = lam * grad_T / np.maximum(T_f, 1e-300)
    K_u = lam * grad_u / (speed + a + 1e-300)
    K_local = np.maximum.reduce([K_rho, K_T, K_u])

    K_rho = np.where(body, np.nan, K_rho)
    K_T = np.where(body, np.nan, K_T)
    K_u = np.where(body, np.nan, K_u)
    K_local = np.where(body, np.nan, K_local)

    Wf_default = wf_loggauss(K_local, 0.1, 1.0)
    Wf_fitted = wf_loggauss(K_local, 0.03, 2.5)

    # Area fractions outside the body.
    fluid = np.isfinite(K_local)
    area_total = np.sum(fluid) * dx * dy
    area_K_gt_0p03 = np.sum((K_local > 0.03) & fluid) * dx * dy
    area_K_gt_0p1 = np.sum((K_local > 0.1) & fluid) * dx * dy
    area_Wf_def_gt_0p5 = np.sum((Wf_default > 0.5) & fluid) * dx * dy
    area_Wf_fit_gt_0p5 = np.sum((Wf_fitted > 0.5) & fluid) * dx * dy

    metrics = {
        "Mach": Mach,
        "Kn": Kn,
        "max_K_local": float(np.nanmax(K_local)),
        "max_K_rho": float(np.nanmax(K_rho)),
        "max_K_T": float(np.nanmax(K_T)),
        "max_K_u": float(np.nanmax(K_u)),
        "max_Wf_default": float(np.nanmax(Wf_default)),
        "max_Wf_fitted": float(np.nanmax(Wf_fitted)),
        "area_total": float(area_total),
        "area_K_gt_0p03": float(area_K_gt_0p03),
        "area_K_gt_0p1": float(area_K_gt_0p1),
        "area_Wf_default_gt_0p5": float(area_Wf_def_gt_0p5),
        "area_Wf_fitted_gt_0p5": float(area_Wf_fit_gt_0p5),
        "frac_K_gt_0p03": float(area_K_gt_0p03 / area_total),
        "frac_K_gt_0p1": float(area_K_gt_0p1 / area_total),
        "frac_Wf_default_gt_0p5": float(area_Wf_def_gt_0p5 / area_total),
        "frac_Wf_fitted_gt_0p5": float(area_Wf_fit_gt_0p5 / area_total),
    }

    field = pd.DataFrame(
        {
            "Mach": Mach,
            "Kn": Kn,
            "x": X.ravel(),
            "y": Y.ravel(),
            "rho": rho.ravel(),
            "T": T.ravel(),
            "ux": ux.ravel(),
            "uy": uy.ravel(),
            "K_local": K_local.ravel(),
            "Wf_default": Wf_default.ravel(),
            "Wf_fitted": Wf_fitted.ravel(),
            "body": body.ravel(),
        }
    )

    arrays = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "body": body,
        "rho": rho,
        "T": T,
        "K_local": K_local,
        "Wf_default": Wf_default,
        "Wf_fitted": Wf_fitted,
    }

    return field, metrics, arrays


def plot_selected(arrays, Mach, Kn):
    X = arrays["X"]
    Y = arrays["Y"]
    body = arrays["body"]

    def masked(A):
        return np.ma.array(A, mask=body | ~np.isfinite(A))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for ax, A, title in [
        (axes[0], arrays["rho"], r"$\rho$"),
        (axes[1], arrays["T"], r"$T$"),
        (axes[2], np.log10(np.maximum(arrays["K_local"], 1e-8)), r"$\log_{10}K_{\rm local}$"),
    ]:
        im = ax.contourf(X, Y, masked(A), levels=40)
        ax.contour(X, Y, body.astype(float), levels=[0.5], linewidths=1.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax)

    axes[0].set_ylabel("y")
    fig.suptitle(f"2D blunt-body diagnostic fields, M={Mach:g}, Kn={Kn:g}")
    fig.savefig(OUT_FIG / "fig51_exp20_2d_blunt_body_fields.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), constrained_layout=True)
    for ax, A, title in [
        (axes[0], arrays["Wf_default"], r"$W_f$ default"),
        (axes[1], arrays["Wf_fitted"], r"$W_f$ fitted"),
    ]:
        im = ax.contourf(X, Y, masked(A), levels=np.linspace(0.0, 1.0, 41))
        ax.contour(X, Y, body.astype(float), levels=[0.5], linewidths=1.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(im, ax=ax)

    axes[0].set_ylabel("y")
    fig.suptitle(f"2D kinetic activation weights, M={Mach:g}, Kn={Kn:g}")
    fig.savefig(OUT_FIG / "fig52_exp20_2d_blunt_body_weights.png", dpi=220)
    plt.close(fig)


def write_table(metrics_df):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Two-dimensional blunt-body shock-layer diagnostic. The field is a smooth analytic bow-shock construction used to test the local rarefaction indicator in curved shock geometry. It is not a full hypersonic CFD or DSMC validation.}",
        r"\label{tab:exp20_blunt_body}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"$M_\infty$ & Kn & $\max K_{\rm local}$ & $\max W_f$ def. & $\max W_f$ fit. & Area $K>0.1$ & Area $W_f>0.5$ fit. \\",
        r"\midrule",
    ]

    for _, r in metrics_df.iterrows():
        lines.append(
            f"{r['Mach']:.1f} & {r['Kn']:.3f} & {r['max_K_local']:.4f} & "
            f"{r['max_Wf_default']:.4f} & {r['max_Wf_fitted']:.4f} & "
            f"{r['frac_K_gt_0p1']:.4f} & {r['frac_Wf_fitted_gt_0p5']:.4f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (OUT_TAB / "table_exp20_2d_blunt_body_metrics.tex").write_text("\n".join(lines) + "\n")


def main():
    cases = [
        (5.0, 0.01),
        (5.0, 0.03),
        (5.0, 0.10),
        (8.0, 0.03),
    ]

    fields = []
    metrics = []
    selected_arrays = None

    for Mach, Kn in cases:
        print(f"case M={Mach:g}, Kn={Kn:g}")
        field, met, arrays = make_case(Mach=Mach, Kn=Kn)
        fields.append(field)
        metrics.append(met)
        if Mach == 5.0 and abs(Kn - 0.03) < 1e-12:
            selected_arrays = arrays
        print(
            f"  maxK={met['max_K_local']:.3e} "
            f"Wf_def={met['max_Wf_default']:.3f} "
            f"Wf_fit={met['max_Wf_fitted']:.3f} "
            f"frac_K>0.1={met['frac_K_gt_0p1']:.3f}"
        )

    df_fields = pd.concat(fields, ignore_index=True)
    df_metrics = pd.DataFrame(metrics)

    df_metrics.to_csv(OUT_DATA / "exp20_2d_blunt_body_metrics.csv", index=False)
    # Store fields too, but round to reduce size.
    df_fields.round(8).to_csv(OUT_DATA / "exp20_2d_blunt_body_fields.csv", index=False)

    if selected_arrays is not None:
        plot_selected(selected_arrays, Mach=5.0, Kn=0.03)

    write_table(df_metrics)

    print("Wrote:")
    print(" ", OUT_DATA / "exp20_2d_blunt_body_metrics.csv")
    print(" ", OUT_DATA / "exp20_2d_blunt_body_fields.csv")
    print(" ", OUT_FIG / "fig51_exp20_2d_blunt_body_fields.png")
    print(" ", OUT_FIG / "fig52_exp20_2d_blunt_body_weights.png")
    print(" ", OUT_TAB / "table_exp20_2d_blunt_body_metrics.tex")
    print()
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
