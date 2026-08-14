#!/usr/bin/env python3
"""
exp18_extended_parameter_calibration.py

Purpose
-------
Extended parameter calibration analysis for the log-Gaussian limiter.

This script uses the existing exp11 raw parameter-scan table and adds:
1. case-weighted parameter ranking;
2. bootstrap robustness of the best parameter pair;
3. per-case best-parameter diagnostics;
4. journal-ready figures and tables.

It does not create new DVM/BGK reference solutions. It is a statistical
robustness analysis over the currently available DVM/BGK benchmark family.

Input
-----
results/data/exp11_loggaussian_parameter_fit_raw.csv

Outputs
-------
results/data/exp18_parameter_case_weighted_summary.csv
results/data/exp18_parameter_bootstrap_counts.csv
results/data/exp18_per_case_best_parameters.csv
results/tables/table_exp18_parameter_robustness.tex
results/figures/fig45_exp18_case_weighted_heatmap.png
results/figures/fig46_exp18_bootstrap_parameter_counts.png
results/figures/fig47_exp18_per_case_best_parameters.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_TABLE = Path("results/tables")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TABLE.mkdir(parents=True, exist_ok=True)


def load_raw():
    path = OUT_DATA / "exp11_loggaussian_parameter_fit_raw.csv"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run python experiments/exp11_fit_loggaussian_parameters.py first."
        )
    df = pd.read_csv(path)

    required = {"case", "Kn", "K0", "sigma", "relative_profile_error"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}. Available columns: {list(df.columns)}")

    df = df.copy()
    df["case_id"] = df["case"].astype(str) + "|Kn=" + df["Kn"].map(lambda x: f"{x:g}")
    return df


def case_weighted_summary(df):
    """
    First average duplicate rows within each case_id and parameter pair,
    then average equally over cases.
    """
    per_case = (
        df.groupby(["case_id", "case", "Kn", "K0", "sigma"], as_index=False)
        .agg(error=("relative_profile_error", "mean"))
    )

    summary = (
        per_case.groupby(["K0", "sigma"], as_index=False)
        .agg(
            mean_error=("error", "mean"),
            median_error=("error", "median"),
            max_error=("error", "max"),
            std_error=("error", "std"),
            n_cases=("error", "size"),
        )
        .sort_values(["mean_error", "max_error", "median_error"])
    )
    return per_case, summary


def bootstrap_best(per_case, n_boot=5000, seed=12345):
    rng = np.random.default_rng(seed)
    case_ids = sorted(per_case["case_id"].unique())
    params = per_case[["K0", "sigma"]].drop_duplicates().reset_index(drop=True)

    # Pivot: rows = case_id, columns = param tuple, values = error
    pivot = per_case.pivot_table(
        index="case_id",
        columns=["K0", "sigma"],
        values="error",
        aggfunc="mean",
    )

    rows = []
    for _ in range(n_boot):
        sample_cases = rng.choice(case_ids, size=len(case_ids), replace=True)
        sample = pivot.loc[sample_cases]
        means = sample.mean(axis=0)
        best_param = means.sort_values().index[0]
        rows.append({"K0": float(best_param[0]), "sigma": float(best_param[1])})

    counts = (
        pd.DataFrame(rows)
        .value_counts(["K0", "sigma"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts["fraction"] = counts["count"] / n_boot
    return counts


def per_case_best(per_case):
    rows = []
    for case_id, sub in per_case.groupby("case_id"):
        best = sub.sort_values(["error", "K0", "sigma"]).iloc[0]
        default = sub[(np.isclose(sub["K0"], 0.1)) & (np.isclose(sub["sigma"], 1.0))]
        fitted = sub[(np.isclose(sub["K0"], 0.03)) & (np.isclose(sub["sigma"], 2.5))]

        rows.append(
            {
                "case_id": case_id,
                "case": best["case"],
                "Kn": float(best["Kn"]),
                "best_K0": float(best["K0"]),
                "best_sigma": float(best["sigma"]),
                "best_error": float(best["error"]),
                "default_error": float(default["error"].iloc[0]) if len(default) else np.nan,
                "fitted_error": float(fitted["error"].iloc[0]) if len(fitted) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["fitted_minus_best"] = out["fitted_error"] - out["best_error"]
    out["default_minus_best"] = out["default_error"] - out["best_error"]
    return out.sort_values(["case", "Kn"])


def make_heatmap(summary):
    piv = summary.pivot(index="sigma", columns="K0", values="mean_error")
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    im = ax.imshow(piv.values, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{x:g}" for x in piv.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([f"{x:g}" for x in piv.index])
    ax.set_xlabel(r"$K_0$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_title("Case-weighted mean profile error")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("mean relative error")

    best = summary.iloc[0]
    x_idx = list(piv.columns).index(best["K0"])
    y_idx = list(piv.index).index(best["sigma"])
    ax.plot([x_idx], [y_idx], marker="*", markersize=14)

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig45_exp18_case_weighted_heatmap.png", dpi=220)
    plt.close(fig)


def make_bootstrap_plot(counts, topn=12):
    top = counts.head(topn).copy()
    labels = [f"K0={r.K0:g}\ns={r.sigma:g}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(np.arange(len(top)), top["fraction"])
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("bootstrap selection fraction")
    ax.set_title("Bootstrap robustness of selected parameters")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig46_exp18_bootstrap_parameter_counts.png", dpi=220)
    plt.close(fig)


def make_per_case_plot(pcb):
    labels = pcb["case_id"].tolist()
    x = np.arange(len(pcb))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.semilogy(x, pcb["default_error"], "o--", label="default")
    ax.semilogy(x, pcb["fitted_error"], "s-", label=r"fitted $K_0=0.03,\sigma=2.5$")
    ax.semilogy(x, pcb["best_error"], "^-", label="per-case best")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("relative profile error")
    ax.set_title("Default, fitted, and per-case best errors")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig47_exp18_per_case_best_parameters.png", dpi=220)
    plt.close(fig)


def write_latex_table(summary, counts, pcb):
    best = summary.iloc[0]
    default = summary[(np.isclose(summary["K0"], 0.1)) & (np.isclose(summary["sigma"], 1.0))]
    fitted = summary[(np.isclose(summary["K0"], 0.03)) & (np.isclose(summary["sigma"], 2.5))]
    top_boot = counts.iloc[0]

    default_mean = float(default["mean_error"].iloc[0]) if len(default) else np.nan
    fitted_mean = float(fitted["mean_error"].iloc[0]) if len(fitted) else np.nan

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Extended parameter calibration robustness analysis based on the available DVM/BGK benchmark family. Case-weighted errors average each physical case equally. Bootstrap fractions are obtained by resampling cases with replacement.}",
        r"\label{tab:exp18_parameter_robustness}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Quantity & Value & Comment \\",
        r"\midrule",
        f"Case-weighted best $K_0$ & {best['K0']:.4g} & minimum mean error \\\\",
        f"Case-weighted best $\\sigma$ & {best['sigma']:.4g} & minimum mean error \\\\",
        f"Best mean error & {best['mean_error']:.4e} & case-weighted \\\\",
        f"Default mean error & {default_mean:.4e} & $K_0=0.1,\\sigma=1.0$ \\\\",
        f"Fitted mean error & {fitted_mean:.4e} & $K_0=0.03,\\sigma=2.5$ \\\\",
        f"Top bootstrap pair & $K_0={top_boot['K0']:.4g},\\sigma={top_boot['sigma']:.4g}$ & selected most often \\\\",
        f"Top bootstrap fraction & {top_boot['fraction']:.3f} & resampled cases \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_TABLE / "table_exp18_parameter_robustness.tex").write_text("\n".join(lines) + "\n")


def main():
    df = load_raw()
    per_case, summary = case_weighted_summary(df)
    counts = bootstrap_best(per_case, n_boot=5000)
    pcb = per_case_best(per_case)

    summary.to_csv(OUT_DATA / "exp18_parameter_case_weighted_summary.csv", index=False)
    counts.to_csv(OUT_DATA / "exp18_parameter_bootstrap_counts.csv", index=False)
    pcb.to_csv(OUT_DATA / "exp18_per_case_best_parameters.csv", index=False)

    make_heatmap(summary)
    make_bootstrap_plot(counts)
    make_per_case_plot(pcb)
    write_latex_table(summary, counts, pcb)

    print("Wrote:")
    print(" ", OUT_DATA / "exp18_parameter_case_weighted_summary.csv")
    print(" ", OUT_DATA / "exp18_parameter_bootstrap_counts.csv")
    print(" ", OUT_DATA / "exp18_per_case_best_parameters.csv")
    print(" ", OUT_TABLE / "table_exp18_parameter_robustness.tex")
    print(" ", OUT_FIG / "fig45_exp18_case_weighted_heatmap.png")
    print(" ", OUT_FIG / "fig46_exp18_bootstrap_parameter_counts.png")
    print(" ", OUT_FIG / "fig47_exp18_per_case_best_parameters.png")
    print()
    print("Top case-weighted parameters:")
    print(summary.head(10).to_string(index=False))
    print()
    print("Top bootstrap counts:")
    print(counts.head(10).to_string(index=False))
    print()
    print("Per-case best:")
    print(pcb.to_string(index=False))


if __name__ == "__main__":
    main()
