"""Compare default and fitted log-Gaussian parameters against DVM/BGK profiles."""

import sys
from pathlib import Path
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erfc

OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

TCOLD = 1.0
THOT = 1.5
UWALL = 0.2

B_TEMP_JUMP = 2.18
A_VELOCITY_SLIP = 1.146


def wf_log(Kn, k0, sigma):
    return float(0.5 * erfc(-np.log(Kn / k0) / (np.sqrt(2.0) * sigma)))


def fourier_nsf(y):
    return TCOLD + (THOT - TCOLD) * y


def fourier_jump(y, Kn):
    return TCOLD + (THOT - TCOLD) * (y + B_TEMP_JUMP * Kn) / (1.0 + 2.0 * B_TEMP_JUMP * Kn)


def couette_nsf(y):
    return UWALL * y


def couette_slip(y, Kn):
    return UWALL * (y + A_VELOCITY_SLIP * Kn) / (1.0 + 2.0 * A_VELOCITY_SLIP * Kn)


def rel_l2(model, ref):
    return float(np.linalg.norm(model - ref) / max(np.linalg.norm(ref), 1e-14))


def main():
    fourier_ref = pd.read_csv(OUT_DATA / "exp09_dvm_bgk_fourier_reference.csv")
    couette_ref = pd.read_csv(OUT_DATA / "exp10_dvm_bgk_couette_reference.csv")

    choices = [
        ("default", 0.1, 1.0),
        ("fitted", 0.03, 2.5),
    ]

    rows = []

    # Fourier profiles
    Kn_values = sorted(fourier_ref["Kn"].unique())
    fig, axes = plt.subplots(1, len(Kn_values), figsize=(13.5, 4.2), sharey=True)

    for ax, Kn in zip(axes, Kn_values):
        s = fourier_ref[fourier_ref["Kn"] == Kn].sort_values("y")
        y = s["y"].to_numpy()
        T_ref = s["T_dvm"].to_numpy()

        ax.plot(T_ref, y, color="black", linewidth=2.2, label="DVM/BGK")

        ax.plot(fourier_nsf(y), y, "--", label="NSF")
        ax.plot(fourier_jump(y, Kn), y, ":", label="jump branch")

        for name, k0, sigma in choices:
            W = wf_log(Kn, k0, sigma)
            T_model = (1.0 - W) * fourier_nsf(y) + W * fourier_jump(y, Kn)
            ax.plot(T_model, y, linewidth=2.0, label=f"{name}")

            rows.append({
                "case": "Fourier",
                "Kn": Kn,
                "choice": name,
                "K0": k0,
                "sigma": sigma,
                "Wf": W,
                "relative_profile_error": rel_l2(T_model, T_ref),
            })

        ax.set_title(f"Kn={Kn:g}")
        ax.set_xlabel(r"$T/T_0$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$y/H$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("DVM/BGK Fourier: default vs fitted log-Gaussian parameters")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig34_dvm_bgk_fourier_default_vs_fitted_profiles.png", dpi=220)
    plt.close(fig)

    # Couette profiles
    Kn_values = sorted(couette_ref["Kn"].unique())
    fig, axes = plt.subplots(1, len(Kn_values), figsize=(13.5, 4.2), sharey=True)

    for ax, Kn in zip(axes, Kn_values):
        s = couette_ref[couette_ref["Kn"] == Kn].sort_values("y")
        y = s["y"].to_numpy()
        u_ref = s["ux_dvm"].to_numpy()

        ax.plot(u_ref / UWALL, y, color="black", linewidth=2.2, label="DVM/BGK")

        ax.plot(couette_nsf(y) / UWALL, y, "--", label="NSF")
        ax.plot(couette_slip(y, Kn) / UWALL, y, ":", label="slip branch")

        for name, k0, sigma in choices:
            W = wf_log(Kn, k0, sigma)
            u_model = (1.0 - W) * couette_nsf(y) + W * couette_slip(y, Kn)
            ax.plot(u_model / UWALL, y, linewidth=2.0, label=f"{name}")

            rows.append({
                "case": "Couette",
                "Kn": Kn,
                "choice": name,
                "K0": k0,
                "sigma": sigma,
                "Wf": W,
                "relative_profile_error": rel_l2(u_model, u_ref),
            })

        ax.set_title(f"Kn={Kn:g}")
        ax.set_xlabel(r"$u/U_w$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$y/H$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("DVM/BGK Couette: default vs fitted log-Gaussian parameters")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig35_dvm_bgk_couette_default_vs_fitted_profiles.png", dpi=220)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "exp12_default_vs_fitted_profile_errors.csv", index=False)

    # Error figure
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)

    for ax, case in zip(axes, ["Fourier", "Couette"]):
        sub = df[df["case"] == case]
        for choice in ["default", "fitted"]:
            s = sub[sub["choice"] == choice].sort_values("Kn")
            ax.loglog(s["Kn"], s["relative_profile_error"], marker="o", label=choice)
        ax.set_title(case)
        ax.set_xlabel("Kn")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("relative profile error vs DVM/BGK")
    axes[-1].legend()
    fig.suptitle("Default vs fitted parameter errors")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig36_dvm_bgk_default_vs_fitted_errors.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print("  results/data/exp12_default_vs_fitted_profile_errors.csv")
    print("  results/figures/fig34_dvm_bgk_fourier_default_vs_fitted_profiles.png")
    print("  results/figures/fig35_dvm_bgk_couette_default_vs_fitted_profiles.png")
    print("  results/figures/fig36_dvm_bgk_default_vs_fitted_errors.png")


if __name__ == "__main__":
    main()
