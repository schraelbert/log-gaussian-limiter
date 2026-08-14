#!/usr/bin/env python3
"""
exp17_flux_moment_validation.py

Purpose
-------
Validate non-equilibrium flux/moment quantities from existing DVM/BGK benchmarks.

Fourier:
    heat-flux closure error from exp09c.

Couette:
    shear-stress error from exp10.

This experiment quantifies whether profile improvements also transfer to
non-equilibrium moment quantities. It is intentionally diagnostic: it may
show that profile accuracy and flux/moment accuracy are not identical.

Outputs
-------
results/data/exp17_flux_moment_validation_summary.csv
results/data/exp17_flux_moment_validation_grouped.csv
results/figures/fig44_flux_moment_errors.png
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


def read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"missing optional input: {path}")
        return None
    return pd.read_csv(path)


def main():
    rows = []

    f9c = read_csv(OUT_DATA / "exp09c_dvm_bgk_fourier_flux_law_metrics.csv")
    if f9c is not None:
        print("exp09c columns:", list(f9c.columns))
        required = {"Kn", "limiter", "relative_flux_law_error"}
        missing = required - set(f9c.columns)
        if missing:
            print("exp09c missing:", missing)
        else:
            for _, r in f9c.iterrows():
                rows.append(
                    {
                        "benchmark": "Fourier",
                        "moment": "heat_flux",
                        "Kn": float(r["Kn"]),
                        "model": str(r["limiter"]),
                        "relative_error": float(r["relative_flux_law_error"]),
                        "source": "exp09c",
                    }
                )

    f10 = read_csv(OUT_DATA / "exp10_dvm_bgk_couette_metrics.csv")
    if f10 is not None:
        print("exp10 columns:", list(f10.columns))
        required = {"Kn", "limiter", "relative_shear_error_vs_DVM"}
        missing = required - set(f10.columns)
        if missing:
            print("exp10 missing:", missing)
        else:
            for _, r in f10.iterrows():
                rows.append(
                    {
                        "benchmark": "Couette",
                        "moment": "shear_stress",
                        "Kn": float(r["Kn"]),
                        "model": str(r["limiter"]),
                        "relative_error": float(r["relative_shear_error_vs_DVM"]),
                        "source": "exp10",
                    }
                )

        # Also record profile error for comparison if available.
        if {"Kn", "limiter", "relative_L2_u_error_vs_DVM"}.issubset(f10.columns):
            for _, r in f10.iterrows():
                rows.append(
                    {
                        "benchmark": "Couette",
                        "moment": "velocity_profile",
                        "Kn": float(r["Kn"]),
                        "model": str(r["limiter"]),
                        "relative_error": float(r["relative_L2_u_error_vs_DVM"]),
                        "source": "exp10",
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable rows found.")

    df.to_csv(OUT_DATA / "exp17_flux_moment_validation_summary.csv", index=False)

    grouped = (
        df.groupby(["benchmark", "moment", "model"], as_index=False)
        .agg(
            mean_error=("relative_error", "mean"),
            max_error=("relative_error", "max"),
            n=("relative_error", "size"),
        )
        .sort_values(["benchmark", "moment", "mean_error"])
    )
    grouped.to_csv(OUT_DATA / "exp17_flux_moment_validation_grouped.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for (benchmark, moment, model), sub in df.groupby(["benchmark", "moment", "model"]):
        sub = sub.sort_values("Kn")
        label = f"{benchmark} {moment}: {model}"
        ax.loglog(
            sub["Kn"],
            np.maximum(sub["relative_error"], 1e-14),
            "o-",
            label=label,
        )

    ax.set_xlabel("Kn")
    ax.set_ylabel("relative error")
    ax.set_title("DVM/BGK flux and moment diagnostic")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig44_flux_moment_errors.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(" ", OUT_DATA / "exp17_flux_moment_validation_summary.csv")
    print(" ", OUT_DATA / "exp17_flux_moment_validation_grouped.csv")
    print(" ", OUT_FIG / "fig44_flux_moment_errors.png")
    print()
    print("Grouped summary:")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
