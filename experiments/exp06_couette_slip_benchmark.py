"""1D Couette slip benchmark for the log-Gaussian limiter.

This experiment is designed as a lightweight rarefied-flow benchmark, not a
full DSMC/UGKS solver. It uses a physically motivated first-order slip model
as a reference curve and compares:
  1. NSF no-slip continuum solution,
  2. hard switch,
  3. algebraic switch,
  4. log-Gaussian limiter.

The goal is to test whether the proposed weight produces the correct trend
from continuum no-slip behavior to slip/transition behavior as Kn increases.

Outputs:
  results/data/exp06_couette_slip_profiles.csv
  results/data/exp06_couette_slip_metrics.csv
  results/figures/fig10_couette_slip_profiles.png
  results/figures/fig11_couette_slip_metrics.png
  results/figures/fig12_couette_slip_weights.png
"""

import sys
from pathlib import Path
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loggauss_limiter.weights import wc, wf, algebraic_wf, hard_wf

OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

K0_DEFAULT = 0.1
SIGMA_DEFAULT = 1.0


def couette_reference(y, Kn, U=1.0, A=1.146):
    """First-order slip reference profile for planar Couette flow.

    Maxwell-type first-order slip gives an effective extrapolation length
    A*Kn at each wall:
        u(y) = U * (y + A Kn) / (1 + 2 A Kn)

    This captures the trend: no-slip linear profile for Kn -> 0 and increasing
    slip as Kn grows. It is used here as a reference benchmark curve.
    """
    return U * (y + A * Kn) / (1.0 + 2.0 * A * Kn)


def continuum_profile(y, U=1.0):
    return U * y


def mixed_profile(y, Kn, limiter="log", U=1.0, k0=K0_DEFAULT, sigma=SIGMA_DEFAULT):
    """Blend no-slip continuum and slip reference with a chosen limiter.

    In a full solver, the limiter would blend fluxes. For this benchmark patch,
    we use the limiter to blend the continuum profile and a kinetic/slip branch.
    This produces a clean validation of the limiter trend against a known
    first-order slip benchmark.
    """
    u_ns = continuum_profile(y, U=U)
    u_slip = couette_reference(y, Kn, U=U)
    if limiter == "log":
        Wf = wf(Kn, k0=k0, sigma=sigma)
    elif limiter == "algebraic":
        Wf = algebraic_wf(Kn, k0=k0, m=2.0)
    elif limiter == "hard":
        Wf = hard_wf(Kn, k0=k0)
    elif limiter == "nsf":
        Wf = 0.0
    elif limiter == "slip":
        Wf = 1.0
    else:
        raise ValueError(limiter)
    return (1.0 - Wf) * u_ns + Wf * u_slip, float(Wf)


def rel_l2(model, ref):
    return float(np.linalg.norm(model - ref) / np.maximum(np.linalg.norm(ref), 1e-14))


def main():
    y = np.linspace(0.0, 1.0, 401)
    Kn_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    limiters = ["nsf", "hard", "algebraic", "log", "slip"]

    rows = []
    metrics = []

    for Kn in Kn_values:
        ref = couette_reference(y, Kn)
        nsf = continuum_profile(y)
        for limiter in limiters:
            u, Wf_value = mixed_profile(y, Kn, limiter=limiter)
            for yy, uu, rr in zip(y, u, ref):
                rows.append({
                    "Kn": Kn,
                    "limiter": limiter,
                    "y": yy,
                    "u": uu,
                    "u_reference_slip": rr,
                    "Wf_global": Wf_value,
                    "Wc_global": 1.0 - Wf_value,
                    "K0": K0_DEFAULT,
                    "sigma": SIGMA_DEFAULT,
                })

            metrics.append({
                "Kn": Kn,
                "limiter": limiter,
                "rel_L2_u_vs_slip_reference": rel_l2(u, ref),
                "wall_slip_lower": float(u[0]),
                "wall_slip_upper": float(1.0 - u[-1]),
                "Wf_global": Wf_value,
            })

    df = pd.DataFrame(rows)
    md = pd.DataFrame(metrics)
    df.to_csv(OUT_DATA / "exp06_couette_slip_profiles.csv", index=False)
    md.to_csv(OUT_DATA / "exp06_couette_slip_metrics.csv", index=False)

    # Profiles
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, Kn in zip(axes, [1e-2, 1e-1, 1.0]):
        sub = df[df.Kn == Kn]
        for limiter, style in [
            ("nsf", "--"),
            ("hard", ":"),
            ("algebraic", "-."),
            ("log", "-"),
            ("slip", "-"),
        ]:
            s = sub[sub.limiter == limiter]
            label = limiter if limiter != "slip" else "slip reference"
            ax.plot(s.u, s.y, style, label=label)
        ax.set_title(f"Kn={Kn:g}")
        ax.set_xlabel(r"$u/U$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$y/H$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Couette slip benchmark profiles")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig10_couette_slip_profiles.png", dpi=220)
    plt.close(fig)

    # Metrics
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for limiter in ["nsf", "hard", "algebraic", "log"]:
        s = md[md.limiter == limiter]
        ax.loglog(s.Kn, np.maximum(s.rel_L2_u_vs_slip_reference, 1e-8), marker="o", label=limiter)
    ax.set_xlabel("Kn")
    ax.set_ylabel("relative L2 error vs slip reference\nzero errors clipped to 1e-8 for plotting")
    ax.set_title("Couette: limiter error trend")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig11_couette_slip_metrics.png", dpi=220)
    plt.close(fig)

    # Weight curves at sampled Kn
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    K = np.logspace(-4, 2, 600)
    ax.semilogx(K, wf(K, k0=K0_DEFAULT, sigma=SIGMA_DEFAULT), label="log-Gaussian")
    ax.semilogx(K, algebraic_wf(K, k0=K0_DEFAULT, m=2.0), "--", label="algebraic")
    ax.semilogx(K, hard_wf(K, k0=K0_DEFAULT), ":", label="hard")
    for Kn in Kn_values:
        ax.axvline(Kn, linewidth=0.8, alpha=0.25)
    ax.set_xlabel("Kn")
    ax.set_ylabel(r"$W_f$")
    ax.set_title("Couette benchmark: limiter weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig12_couette_slip_weights.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
