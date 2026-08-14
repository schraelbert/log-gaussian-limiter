"""1D discrete-velocity BGK Fourier benchmark.

This experiment computes a simplified kinetic reference for Fourier heat
transfer between two diffuse walls using a stationary 1D-in-space, 2D-in-velocity
BGK model:

    v_y df/dy = (M[f] - f) / tau.

The resulting DVM/BGK temperature profile is then compared with:
  - NSF linear temperature profile,
  - hard-switch slip/jump model,
  - algebraic limiter model,
  - log-Gaussian limiter model.

This is still a compact research prototype, not a production UGKS/DSMC solver,
but it is substantially stronger than purely synthetic slip/jump references.
"""

import sys
from pathlib import Path
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loggauss_limiter.weights import wf, algebraic_wf, hard_wf

OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

D = 2
R = 1.0

TCOLD = 1.0
THOT = 1.5

K0_MODEL = 0.1
SIGMA_MODEL = 1.0


def maxwellian(vx, vy, rho, ux, uy, T):
    T = np.maximum(T, 1.0e-12)
    c2 = (vx - ux) ** 2 + (vy - uy) ** 2
    return rho / (2.0 * np.pi * R * T) * np.exp(-c2 / (2.0 * R * T))


def moments(f, vx, vy, wv):
    rho = np.sum(f * wv, axis=(1, 2))
    rho = np.maximum(rho, 1.0e-14)

    ux = np.sum(f * vx[None, :, :] * wv, axis=(1, 2)) / rho
    uy = np.sum(f * vy[None, :, :] * wv, axis=(1, 2)) / rho

    e = np.sum(0.5 * (vx[None, :, :] ** 2 + vy[None, :, :] ** 2) * f * wv, axis=(1, 2))
    T = (2.0 * e / rho - ux ** 2 - uy ** 2) / D
    T = np.maximum(T, 1.0e-8)
    return rho, ux, uy, T


def heat_flux_y(f, vx, vy, wv, rho, ux, uy, T):
    cx = vx[None, :, :] - ux[:, None, None]
    cy = vy[None, :, :] - uy[:, None, None]
    c2 = cx ** 2 + cy ** 2
    qy = np.sum(0.5 * c2 * cy * f * wv, axis=(1, 2))
    return qy


def diffuse_wall_density_left(f0, vx, vy, wv, Twall):
    """Density of diffuse wall Maxwellian so that net mass flux is zero."""
    out_mask = vy < 0.0
    in_mask = vy > 0.0

    outgoing = np.sum(vy[out_mask] * f0[out_mask] * wv[out_mask])

    M_unit = maxwellian(vx, vy, rho=1.0, ux=0.0, uy=0.0, T=Twall)
    incoming_unit = np.sum(vy[in_mask] * M_unit[in_mask] * wv[in_mask])

    return max(1.0e-12, -outgoing / max(incoming_unit, 1.0e-14))


def diffuse_wall_density_right(fN, vx, vy, wv, Twall):
    """Density of diffuse wall Maxwellian so that net mass flux is zero."""
    out_mask = vy > 0.0
    in_mask = vy < 0.0

    outgoing = np.sum(vy[out_mask] * fN[out_mask] * wv[out_mask])

    M_unit = maxwellian(vx, vy, rho=1.0, ux=0.0, uy=0.0, T=Twall)
    incoming_unit = np.sum(vy[in_mask] * M_unit[in_mask] * wv[in_mask])

    return max(1.0e-12, -outgoing / min(incoming_unit, -1.0e-14))


def solve_dvm_bgk_fourier(Kn, Ny=121, Nv=33, Vmax=7.0, max_iter=2500, tol=3.0e-8):
    y = np.linspace(0.0, 1.0, Ny)
    dy = y[1] - y[0]

    v = np.linspace(-Vmax, Vmax, Nv)
    dv = v[1] - v[0]
    vx, vy = np.meshgrid(v, v, indexing="ij")
    wv = np.full_like(vx, dv * dv)

    T_init = TCOLD + (THOT - TCOLD) * y
    rho_init = np.ones_like(y)
    ux_init = np.zeros_like(y)
    uy_init = np.zeros_like(y)

    f = np.zeros((Ny, Nv, Nv), dtype=float)
    for j in range(Ny):
        f[j] = maxwellian(vx, vy, rho_init[j], ux_init[j], uy_init[j], T_init[j])

    # In this nondimensional prototype tau is proportional to Kn.
    tau = max(Kn, 1.0e-6)

    pos = vy > 1.0e-14
    neg = vy < -1.0e-14
    zero = ~(pos | neg)

    history = []

    for it in range(max_iter):
        f_old = f.copy()

        rho, ux, uy, T = moments(f, vx, vy, wv)

        M = np.zeros_like(f)
        for j in range(Ny):
            M[j] = maxwellian(vx, vy, rho[j], ux[j], uy[j], T[j])

        # Diffuse wall boundaries.
        rho_wall_left = diffuse_wall_density_left(f[0], vx, vy, wv, TCOLD)
        rho_wall_right = diffuse_wall_density_right(f[-1], vx, vy, wv, THOT)

        M_left = maxwellian(vx, vy, rho_wall_left, 0.0, 0.0, TCOLD)
        M_right = maxwellian(vx, vy, rho_wall_right, 0.0, 0.0, THOT)

        f[0, pos] = M_left[pos]
        f[-1, neg] = M_right[neg]

        # Sweep positive velocities left to right.
        for j in range(1, Ny):
            a = vy[pos] / dy
            f[j, pos] = (a * f[j - 1, pos] + M[j, pos] / tau) / (a + 1.0 / tau)

        # Sweep negative velocities right to left.
        for j in range(Ny - 2, -1, -1):
            a = (-vy[neg]) / dy
            f[j, neg] = (a * f[j + 1, neg] + M[j, neg] / tau) / (a + 1.0 / tau)

        # Near-zero vy velocities relax locally.
        f[:, zero] = M[:, zero]

        # Positivity guard.
        f = np.maximum(f, 1.0e-300)

        rel = np.linalg.norm(f - f_old) / max(np.linalg.norm(f_old), 1.0e-14)
        if it % 50 == 0 or it == max_iter - 1:
            history.append((it, rel))
        if rel < tol and it > 50:
            history.append((it, rel))
            break

    rho, ux, uy, T = moments(f, vx, vy, wv)
    qy = heat_flux_y(f, vx, vy, wv, rho, ux, uy, T)

    return {
        "Kn": Kn,
        "y": y,
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "T": T,
        "qy": qy,
        "iterations": it + 1,
        "final_rel_change": rel,
        "history": history,
    }


def fourier_nsf(y):
    return TCOLD + (THOT - TCOLD) * y


def fourier_jump_branch(y, Kn, B=2.18):
    return TCOLD + (THOT - TCOLD) * (y + B * Kn) / (1.0 + 2.0 * B * Kn)


def limiter_weight(Kn, limiter):
    if limiter == "nsf":
        return 0.0
    if limiter == "hard":
        return float(hard_wf(Kn, k0=K0_MODEL))
    if limiter == "algebraic":
        return float(algebraic_wf(Kn, k0=K0_MODEL, m=2.0))
    if limiter == "log":
        return float(wf(Kn, k0=K0_MODEL, sigma=SIGMA_MODEL))
    if limiter == "jump":
        return 1.0
    raise ValueError(limiter)


def model_temperature(y, Kn, limiter):
    T_nsf = fourier_nsf(y)
    T_jump = fourier_jump_branch(y, Kn)
    W = limiter_weight(Kn, limiter)
    return (1.0 - W) * T_nsf + W * T_jump, W


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-14))


def mean_heat_flux_from_profile(y, T, kappa=1.0):
    return -kappa * float(np.mean(np.gradient(T, y)[10:-10]))


def main():
    Kn_values = [1.0e-2, 1.0e-1, 1.0]
    limiters = ["nsf", "hard", "algebraic", "log", "jump"]

    ref_rows = []
    model_rows = []
    metrics = []
    conv_rows = []

    for Kn in Kn_values:
        print(f"Solving DVM/BGK Fourier reference, Kn={Kn:g}")
        sol = solve_dvm_bgk_fourier(Kn)

        y = sol["y"]
        T_ref = sol["T"]
        q_ref = float(np.mean(sol["qy"][10:-10]))

        for it, rel in sol["history"]:
            conv_rows.append({
                "Kn": Kn,
                "iteration": it,
                "relative_change": rel,
                "final_iterations": sol["iterations"],
                "final_rel_change": sol["final_rel_change"],
            })

        for j in range(len(y)):
            ref_rows.append({
                "Kn": Kn,
                "y": y[j],
                "rho_dvm": sol["rho"][j],
                "ux_dvm": sol["ux"][j],
                "uy_dvm": sol["uy"][j],
                "T_dvm": T_ref[j],
                "qy_dvm": sol["qy"][j],
                "iterations": sol["iterations"],
                "final_rel_change": sol["final_rel_change"],
            })

        for limiter in limiters:
            T_model, W = model_temperature(y, Kn, limiter)
            q_model = mean_heat_flux_from_profile(y, T_model)

            for yy, tm, tr in zip(y, T_model, T_ref):
                model_rows.append({
                    "Kn": Kn,
                    "limiter": limiter,
                    "y": yy,
                    "T_model": tm,
                    "T_dvm_reference": tr,
                    "Wf": W,
                    "K0_model": K0_MODEL,
                    "sigma_model": SIGMA_MODEL,
                })

            metrics.append({
                "Kn": Kn,
                "limiter": limiter,
                "Wf": W,
                "relative_L2_T_error_vs_DVM": rel_l2(T_model, T_ref),
                "q_model": q_model,
                "q_DVM_reference": q_ref,
                "relative_heat_flux_error_vs_DVM": abs(q_model - q_ref) / max(abs(q_ref), 1.0e-14),
                "T_lower_model": float(T_model[0]),
                "T_upper_model": float(T_model[-1]),
                "T_lower_DVM": float(T_ref[0]),
                "T_upper_DVM": float(T_ref[-1]),
            })

    df_ref = pd.DataFrame(ref_rows)
    df_model = pd.DataFrame(model_rows)
    df_metrics = pd.DataFrame(metrics)
    df_conv = pd.DataFrame(conv_rows)

    df_ref.to_csv(OUT_DATA / "exp09_dvm_bgk_fourier_reference.csv", index=False)
    df_model.to_csv(OUT_DATA / "exp09_dvm_bgk_fourier_models.csv", index=False)
    df_metrics.to_csv(OUT_DATA / "exp09_dvm_bgk_fourier_metrics.csv", index=False)
    df_conv.to_csv(OUT_DATA / "exp09_dvm_bgk_fourier_convergence.csv", index=False)

    # Temperature profiles.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, Kn in zip(axes, Kn_values):
        ref = df_ref[df_ref["Kn"] == Kn]
        ax.plot(ref["T_dvm"], ref["y"], color="black", linewidth=2.0, label="DVM/BGK reference")

        sub = df_model[df_model["Kn"] == Kn]
        for limiter, style in [
            ("nsf", "--"),
            ("hard", ":"),
            ("algebraic", "-."),
            ("log", "-"),
            ("jump", "--"),
        ]:
            s = sub[sub["limiter"] == limiter]
            ax.plot(s["T_model"], s["y"], style, label=limiter)

        ax.set_title(f"Kn={Kn:g}")
        ax.set_xlabel(r"$T/T_0$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$y/H$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("DVM/BGK Fourier benchmark: temperature profiles")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig20_dvm_bgk_fourier_profiles.png", dpi=220)
    plt.close(fig)

    # Error vs DVM.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for limiter in ["nsf", "hard", "algebraic", "log", "jump"]:
        s = df_metrics[df_metrics["limiter"] == limiter]
        axes[0].loglog(
            s["Kn"],
            np.maximum(s["relative_L2_T_error_vs_DVM"], 1.0e-12),
            marker="o",
            label=limiter,
        )
        axes[1].loglog(
            s["Kn"],
            np.maximum(s["relative_heat_flux_error_vs_DVM"], 1.0e-12),
            marker="o",
            label=limiter,
        )

    axes[0].set_xlabel("Kn")
    axes[0].set_ylabel("relative L2 T error vs DVM/BGK")
    axes[0].set_title("Temperature profile error")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].set_xlabel("Kn")
    axes[1].set_ylabel("relative heat-flux error vs DVM/BGK")
    axes[1].set_title("Heat-flux error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.suptitle("DVM/BGK Fourier benchmark: model comparison")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig21_dvm_bgk_fourier_errors.png", dpi=220)
    plt.close(fig)

    # DVM heat flux and convergence.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for Kn in Kn_values:
        ref = df_ref[df_ref["Kn"] == Kn]
        ax.plot(ref["y"], ref["qy_dvm"], label=f"Kn={Kn:g}")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$q_y$ from DVM/BGK")
    ax.set_title("DVM/BGK Fourier heat flux")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig22_dvm_bgk_fourier_heat_flux.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for Kn in Kn_values:
        c = df_conv[df_conv["Kn"] == Kn]
        ax.semilogy(c["iteration"], c["relative_change"], marker="o", label=f"Kn={Kn:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative change")
    ax.set_title("DVM/BGK source-iteration convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig23_dvm_bgk_fourier_convergence.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print("  results/data/exp09_dvm_bgk_fourier_reference.csv")
    print("  results/data/exp09_dvm_bgk_fourier_models.csv")
    print("  results/data/exp09_dvm_bgk_fourier_metrics.csv")
    print("  results/data/exp09_dvm_bgk_fourier_convergence.csv")
    print("  results/figures/fig20_dvm_bgk_fourier_profiles.png")
    print("  results/figures/fig21_dvm_bgk_fourier_errors.png")
    print("  results/figures/fig22_dvm_bgk_fourier_heat_flux.png")
    print("  results/figures/fig23_dvm_bgk_fourier_convergence.png")


if __name__ == "__main__":
    main()
