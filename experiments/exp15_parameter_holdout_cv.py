#!/usr/bin/env python3
"""
exp15_parameter_holdout_cv.py

Purpose
-------
Use the existing parameter-scan data from exp11 to perform a simple
leave-one-case-out cross-validation study.

This script does not replace independent validation, but it separates
training and held-out cases within the available DVM/BGK benchmark set.

Expected input
--------------
results/data/exp11_loggaussian_parameter_fit_raw.csv

The script is deliberately robust to column-name variations.

Outputs
-------
results/data/exp15_parameter_holdout_cv.csv
results/data/exp15_parameter_holdout_cv_summary.csv
results/figures/fig40_parameter_holdout_cv.png
results/figures/fig41_parameter_holdout_selected_params.png
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DATA = Path("results/data")
OUT_FIG = Path("results/figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)


def find_col(df, candidates, required=True):
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"Could not find any of columns {candidates}. Available columns: {list(df.columns)}")
    return None


def load_raw_scan():
    path = OUT_DATA / "exp11_loggaussian_parameter_fit_raw.csv"
    if not path.exists():
        print(f"Missing {path}")
        print("Run exp11 first:")
        print("  python experiments/exp11_fit_loggaussian_parameters.py")
        sys.exit(2)

    df = pd.read_csv(path)
    return df


def normalize_scan_df(df):
    k0_col = find_col(df, ["K0", "k0", "K_0"])
    sigma_col = find_col(df, ["sigma", "Sigma"])
    err_col = find_col(df, ["error", "relative_error", "relative_profile_error", "profile_error", "E", "E_phi", "err"])

    case_col = find_col(df, ["case", "benchmark", "flow", "problem"], required=False)
    kn_col = find_col(df, ["Kn", "kn", "Kn_global", "K"], required=False)
    quantity_col = find_col(df, ["quantity", "phi", "variable"], required=False)

    out = pd.DataFrame()
    out["K0"] = df[k0_col].astype(float)
    out["sigma"] = df[sigma_col].astype(float)
    out["error"] = df[err_col].astype(float)

    if case_col is None:
        out["case"] = "case"
    else:
        out["case"] = df[case_col].astype(str)

    if kn_col is None:
        out["Kn"] = np.nan
    else:
        out["Kn"] = df[kn_col].astype(float)

    if quantity_col is None:
        out["quantity"] = ""
    else:
        out["quantity"] = df[quantity_col].astype(str)

    # Define a held-out unit. This tries to keep one physical profile/case out.
    out["holdout_id"] = (
        out["case"].astype(str)
        + "|Kn="
        + out["Kn"].map(lambda x: "nan" if pd.isna(x) else f"{x:g}")
        + "|"
        + out["quantity"].astype(str)
    )

    return out


def main():
    raw = load_raw_scan()
    df = normalize_scan_df(raw)

    param_cols = ["K0", "sigma"]
    params = df[param_cols].drop_duplicates().sort_values(param_cols).reset_index(drop=True)
    holdouts = sorted(df["holdout_id"].unique())

    rows = []

    for h in holdouts:
        train = df[df["holdout_id"] != h]
        test = df[df["holdout_id"] == h]

        if train.empty or test.empty:
            continue

        train_mean = (
            train.groupby(param_cols, as_index=False)["error"]
            .mean()
            .rename(columns={"error": "train_mean_error"})
        )

        best = train_mean.sort_values("train_mean_error", ascending=True).iloc[0]
        K0_best = float(best["K0"])
        sigma_best = float(best["sigma"])

        test_at_best = test[(test["K0"] == K0_best) & (test["sigma"] == sigma_best)]
        if test_at_best.empty:
            # Should not happen if full grid exists.
            continue

        # Default and full-data best for comparison.
        test_default = test[(np.isclose(test["K0"], 0.1)) & (np.isclose(test["sigma"], 1.0))]
        if test_default.empty:
            default_error = np.nan
        else:
            default_error = float(test_default["error"].mean())

        full_mean = (
            df.groupby(param_cols, as_index=False)["error"]
            .mean()
            .rename(columns={"error": "full_mean_error"})
        )
        full_best = full_mean.sort_values("full_mean_error", ascending=True).iloc[0]
        K0_full = float(full_best["K0"])
        sigma_full = float(full_best["sigma"])

        test_full_best = test[(test["K0"] == K0_full) & (test["sigma"] == sigma_full)]
        full_best_test_error = float(test_full_best["error"].mean()) if not test_full_best.empty else np.nan

        rows.append(
            {
                "holdout_id": h,
                "selected_K0": K0_best,
                "selected_sigma": sigma_best,
                "train_mean_error": float(best["train_mean_error"]),
                "test_error_selected": float(test_at_best["error"].mean()),
                "test_error_default": default_error,
                "full_data_best_K0": K0_full,
                "full_data_best_sigma": sigma_full,
                "test_error_full_data_best": full_best_test_error,
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
            }
        )

    cv = pd.DataFrame(rows)
    if cv.empty:
        raise RuntimeError("No CV rows produced. Check exp11 raw CSV structure.")

    cv.to_csv(OUT_DATA / "exp15_parameter_holdout_cv.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "mean_test_error_selected_by_holdout_training",
                "value": cv["test_error_selected"].mean(),
            },
            {
                "metric": "mean_test_error_default",
                "value": cv["test_error_default"].mean(),
            },
            {
                "metric": "mean_test_error_full_data_best",
                "value": cv["test_error_full_data_best"].mean(),
            },
            {
                "metric": "median_selected_K0",
                "value": cv["selected_K0"].median(),
            },
            {
                "metric": "median_selected_sigma",
                "value": cv["selected_sigma"].median(),
            },
            {
                "metric": "n_holdouts",
                "value": len(cv),
            },
        ]
    )
    summary.to_csv(OUT_DATA / "exp15_parameter_holdout_cv_summary.csv", index=False)

    # Figure 40: held-out errors.
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(cv))
    ax.plot(x, cv["test_error_default"], "o--", label="default")
    ax.plot(x, cv["test_error_selected"], "s-", label="selected by training")
    ax.plot(x, cv["test_error_full_data_best"], "^-", label="full-data best")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(cv["holdout_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("held-out relative profile error")
    ax.set_title("Leave-one-case-out parameter check")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig40_parameter_holdout_cv.png", dpi=220)
    plt.close(fig)

    # Figure 41: selected parameters by holdout.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(
        cv["selected_K0"],
        cv["selected_sigma"],
        c=cv["test_error_selected"],
        s=90,
        edgecolor="k",
    )
    ax.scatter([0.1], [1.0], marker="x", s=100, label="default")
    if not cv["full_data_best_K0"].isna().all():
        ax.scatter(
            [cv["full_data_best_K0"].iloc[0]],
            [cv["full_data_best_sigma"].iloc[0]],
            marker="*",
            s=160,
            label="full-data best",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"selected $K_0$")
    ax.set_ylabel(r"selected $\sigma$")
    ax.set_title("Parameters selected by holdout training")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("held-out error")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig41_parameter_holdout_selected_params.png", dpi=220)
    plt.close(fig)

    print("Wrote:")
    print(" ", OUT_DATA / "exp15_parameter_holdout_cv.csv")
    print(" ", OUT_DATA / "exp15_parameter_holdout_cv_summary.csv")
    print(" ", OUT_FIG / "fig40_parameter_holdout_cv.png")
    print(" ", OUT_FIG / "fig41_parameter_holdout_selected_params.png")
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print("Holdout rows:")
    print(cv.to_string(index=False))


if __name__ == "__main__":
    main()
