"""1D discrete-velocity BGK Couette benchmark.

Stationary 1D-in-space, 2D-in-velocity BGK model:

    v_y df/dy = (M[f] - f) / tau.

Diffuse walls:
    lower wall: ux = 0
    upper wall: ux = Uwall

The DVM/BGK velocity profile is used as a kinetic reference. We compare:
  - NSF no-slip profile,
  - hard-switch slip model,
  - algebraic limiter model,
  - log-Gaussian limiter model,
  - pure slip branch.

Outputs:
  results/data/exp10_dvm_bgk_couette_reference.csv
  results/data/exp10_dvm_bgk_couette_models.csv
  results/data/exp10_dvm_bgk_couette_metrics.csv
  results/data/exp10_dvm_bgk_couette_convergence.csv
  results/figures/fig26_dvm_bgk_couette_profiles.png
  results/figures/fig27_dvm_bgk_couette_errors.png
  results/figures/fig28_dvm_bgk_couette_shear.png
  results/figures/fig29_dvm_bgk_couette_convergence.png
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

TWALL = 1.0
UWALL = 0.2

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


def shear_stress_xy(f, vx, vy, wv, rho, ux, uy):
    cx = vx[None, :, :] - ux[:, None, None]
    cy = vy[None, :, :] - uy[:, None, None]
    return np.sum(cx * cy * f * wv, axis=(1, 2))


def diffuse_wall_density_left(f0, vx, vy, wv, ux_wall, T_wall):
    out_mask = vy < 0.0
    in_mask = vy > 0.0

    outgoing = np.sum(vy[out_mask] * f0[out_mask] * wv[out_mask])
    M_unit = maxwellian(vx, vy, rho=1.0, ux=ux_wall, uy=0.0, T=T_wall)
    incoming_unit = np.sum(vy[in_mask] * M_unit[in_mask] * wv[in_mask])

    return max(1.0e-12, -outgoing / max(incoming_unit, 1.0e-14))


def diffuse_wall_density_right(fN, vx, vy, wv, ux_wall, T_wall):
    out_mask = vy > 0.0
    in_mask = vy < 0.0

    outgoing = np.sum(vy[out_mask] * fN[out_mask] * wv[out_mask])
    M_unit = maxwellian(vx, vy, rho=1.0, ux=ux_wall, uy=0.0, T=T_wall)
    incoming_unit = np.sum(vy[in_mask] * M_unit[in_mask] * wv[in_mask])

    return max(1.0e-12, -outgoing / min(incoming_unit, -1.0e-14))


def solve_dvm_bgk_couette(Kn, Ny=121, Nv=33, Vmax=7.0, max_iter=2500, tol=3.0e-8):
    y = np.linspace(0.0, 1.0, Ny)
    dy = y[1] - y[0]

    v = np.linspace(-Vmax, Vmax, Nv)
    dv = v[1] - v[0]
    vx, vy = np.meshgrid(v, v, indexing="ij")
    wv = np.full_like(vx, dv * dv)

    ux_init = UWALL * y
    T_init = np.full_like(y, TWALL)
    rho_init = np.ones_like(y)
    uy_init = np.zeros_like(y)

    f = np.zeros((Ny, Nv, Nv), dtype=float)
    for j in range(Ny):
        f[j] = maxwellian(vx, vy, rho_init[j], ux_init[j], uy_init[j], T_init[j])

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

        rho_wall_left = diffuse_wall_density_left(f[0], vx, vy, wv, 0.0, TWALL)
        rho_wall_right = diffuse_wall_density_right(f[-1], vx, vy, wv, UWALL, TWALL)

        M_left = maxwellian(vx, vy, rho_wall_left, 0.0, 0.0, TWALL)
        M_right = maxwellian(vx, vy, rho_wall_right, UWALL, 0.0, TWALL)

        f[0, pos] = M_left[pos]
        f[-1, neg] = M_right[neg]

        for j in range(1, Ny):
            a = vy[pos] / dy
            f[j, pos] = (a * f[j - 1, pos] + M[j, pos] / tau) / (a + 1.0 / tau)

        for j in range(Ny - 2, -1, -1):
            a = (-vy[neg]) / dy
            f[j, neg] = (a * f[j + 1, neg] + M[j, neg] / tau) / (a + 1.0 / tau)

        f[:, zero] = M[:, zero]
        f = np.maximum(f, 1.0e-300)

        rel = np.linalg.norm(f - f_old) / max(np.linalg.norm(f_old), 1.0e-14)
        if it % 50 == 0 or it == max_iter - 1:
            history.append((it, rel))
        if rel < tol and it > 50:
            history.append((it, rel))
            break

    rho, ux, uy, T = moments(f, vx, vy, wv)
    pxy = shear_stress_xy(f, vx, vy, wv, rho, ux, uy)

    return {
        "Kn": Kn,
        "y": y,
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "T": T,
        "pxy": pxy,
        "iterations": it + 1,
        "final_rel_change": rel,
        "history": history,
    }


def couette_nsf(y):
    return UWALL * y


def couette_slip_branch(y, Kn, A=1.146):
    return UWALL * (y + A * Kn) / (1.0 + 2.0 * A * Kn)


def limiter_weight(Kn, limiter):
    if limiter == "nsf":
        return 0.0
    if limiter == "hard":
        return float(hard_wf(Kn, k0=K0_MODEL))
    if limiter == "algebraic":
        return float(algebraic_wf(Kn, k0=K0_MODEL, m=2.0))
    if limiter == "log":
        return float(wf(Kn, k0=K0_MODEL, sigma=SIGMA_MODEL))
    if limiter == "slip":
        return 1.0
    raise ValueError(limiter)


def model_velocity(y, Kn, limiter):
    u_nsf = couette_nsf(y)
    u_slip = couette_slip_branch(y, Kn)
    W = limiter_weight(Kn, limiter)
    return (1.0 - W) * u_nsf + W * u_slip, W


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-14))


def shear_from_profile(y, u, mu_eff=1.0):
    return mu_eff * float(np.mean(np.gradient(u, y)[10:-10]))


def main():
    Kn_values = [1.0e-2, 1.0e-1, 1.0]
    limiters = ["nsf", "hard", "algebraic", "log", "slip"]

    ref_rows = []
    model_rows = []
    metrics = []
    conv_rows = []

    for Kn in Kn_values:
        print(f"Solving DVM/BGK Couette reference, Kn={Kn:g}")
        sol = solve_dvm_bgk_couette(Kn)

        y = sol["y"]
        u_ref = sol["ux"]
        pxy_ref = sol["pxy"]
        shear_ref = float(np.mean(pxy_ref[10:-10]))

        du_ref = float(np.mean(np.gradient(u_ref, y)[10:-10]))
        mu_eff = shear_ref / du_ref if abs(du_ref) > 1.0e-14 else 1.0

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
                "T_dvm": sol["T"][j],
                "pxy_dvm": sol["pxy"][j],
                "iterations": sol["iterations"],
                "final_rel_change": sol["final_rel_change"],
            })

        for limiter in limiters:
            u_model, W = model_velocity(y, Kn, limiter)
            shear_model = shear_from_profile(y, u_model, mu_eff=mu_eff)

            for yy, um, ur in zip(y, u_model, u_ref):
                model_rows.append({
                    "Kn": Kn,
                    "limiter": limiter,
                    "y": yy,
                    "u_model": um,
                    "u_dvm_reference": ur,
                    "Wf": W,
                    "K0_model": K0_MODEL,
                    "sigma_model": SIGMA_MODEL,
                })

            metrics.append({
                "Kn": Kn,
                "limiter": limiter,
                "Wf": W,
                "relative_L2_u_error_vs_DVM": rel_l2(u_model, u_ref),
                "shear_model": shear_model,
                "shear_DVM_reference": shear_ref,
                "relative_shear_error_vs_DVM": abs(shear_model - shear_ref) / max(abs(shear_ref), 1.0e-14),
                "u_lower_model": float(u_model[0]),
                "u_upper_model": float(u_model[-1]),
                "u_lower_DVM": float(u_ref[0]),
                "u_upper_DVM": float(u_ref[-1]),
                "mu_eff_DVM": mu_eff,
            })

    df_ref = pd.DataFrame(ref_rows)
    df_model = pd.DataFrame(model_rows)
    df_metrics = pd.DataFrame(metrics)
    df_conv = pd.DataFrame(conv_rows)

    df_ref.to_csv(OUT_DATA / "exp10_dvm_bgk_couette_reference.csv", index=False)
    df_model.to_csv(OUT_DATA / "exp10_dvm_bgk_couette_models.csv", index=False)
    df_metrics.to_csv(OUT_DATA / "exp10_dvm_bgk_couette_metrics.csv", index=False)
    df_conv.to_csv(OUT_DATA / "exp10_dvm_bgk_couette_convergence.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, Kn in zip(axes, Kn_values):
        ref = df_ref[df_ref["Kn"] == Kn]
        ax.plot(ref["ux_dvm"] / UWALL, ref["y"], color="black", linewidth=2.0, label="DVM/BGK reference")

        sub = df_model[df_model["Kn"] == Kn]
        for limiter, style in [
            ("nsf", "--"),
            ("hard", ":"),
            ("algebraic", "-."),
            ("log", "-"),
            ("slip", "--"),
        ]:
            s = sub[sub["limiter"] == limiter]
            ax.plot(s["u_model"] / UWALL, s["y"], style, label=limiter)

        ax.set_title(f"Kn={Kn:g}")
        ax.set_xlabel(r"$u/U_w$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$y/H$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("DVM/BGK Couette benchmark: velocity profiles")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig26_dvm_bgk_couette_profiles.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for limiter in limiters:
        s = df_metrics[df_metrics["limiter"] == limiter]
        axes[0].loglog(
            s["Kn"],
            np.maximum(s["relative_L2_u_error_vs_DVM"], 1.0e-12),
            marker="o",
            label=limiter,
        )
        axes[1].loglog(
            s["Kn"],
            np.maximum(s["relative_shear_error_vs_DVM"], 1.0e-12),
            marker="o",
            label=limiter,
        )

    axes[0].set_xlabel("Kn")
    axes[0].set_ylabel("relative L2 u error vs DVM/BGK")
    axes[0].set_title("Velocity profile error")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].set_xlabel("Kn")
    axes[1].set_ylabel("relative shear error vs DVM/BGK")
    axes[1].set_title("Shear-stress error")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.suptitle("DVM/BGK Couette benchmark: model comparison")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig27_dvm_bgk_couette_errors.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for Kn in Kn_values:
        ref = df_ref[df_ref["Kn"] == Kn]
        ax.plot(ref["y"], ref["pxy_dvm"], label=f"Kn={Kn:g}")
    ax.set_xlabel(r"$y/H$")
    ax.set_ylabel(r"$P_{xy}$ from DVM/BGK")
    ax.set_title("DVM/BGK Couette shear stress")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig28_dvm_bgk_couette_shear.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for Kn in Kn_values:
        c = df_conv[df_conv["Kn"] == Kn]
        ax.semilogy(c["iteration"], c["relative_change"], marker="o", label=f"Kn={Kn:g}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative change")
    ax.set_title("DVM/BGK Couette source-iteration convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig29_dvm_bgk_couette_convergence.png", dpi=220)
    plt.close(fig)

    print("Wrote exp10 DVM/BGK Couette outputs.")


if __name__ == "__main__":
    main()
