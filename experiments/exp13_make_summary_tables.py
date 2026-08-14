"""Create compact summary tables for the published article."""

from pathlib import Path
import pandas as pd

OUT_DATA = Path("results/data")
OUT_TABLE = Path("results/tables")
OUT_TABLE.mkdir(parents=True, exist_ok=True)

exp11 = pd.read_csv(OUT_DATA / "exp11_default_vs_best_errors.csv")
exp12 = pd.read_csv(OUT_DATA / "exp12_default_vs_fitted_profile_errors.csv")

rows = []

for df_name, df in [("exp11", exp11), ("exp12", exp12)]:
    for choice in sorted(df["choice"].unique()):
        s = df[df["choice"] == choice]
        rows.append({
            "source": df_name,
            "choice": choice,
            "mean_error_all": s["relative_profile_error"].mean(),
            "max_error_all": s["relative_profile_error"].max(),
            "mean_error_Fourier": s[s["case"] == "Fourier"]["relative_profile_error"].mean(),
            "mean_error_Couette": s[s["case"] == "Couette"]["relative_profile_error"].mean(),
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DATA / "exp13_default_vs_fitted_summary.csv", index=False)

article_table = summary[summary["source"] == "exp12"].copy()

def params(choice):
    if choice == "default":
        return r"$K_0=0.1,\quad \sigma=1.0$"
    if choice in ["best_mean", "fitted"]:
        return r"$K_0=0.03,\quad \sigma=2.5$"
    return ""

article_table["parameters"] = article_table["choice"].map(params)

article_table = article_table[
    [
        "choice",
        "parameters",
        "mean_error_Fourier",
        "mean_error_Couette",
        "mean_error_all",
        "max_error_all",
    ]
]

article_table.to_csv(OUT_TABLE / "table_default_vs_fitted.csv", index=False)

def fmt(x):
    return f"{float(x):.4e}"

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Default and DVM/BGK-calibrated log-Gaussian parameters.}")
lines.append(r"\label{tab:default_vs_fitted}")
lines.append(r"\begin{tabular}{llrrrr}")
lines.append(r"\hline")
lines.append(r"Choice & Parameters & Fourier mean & Couette mean & Overall mean & Max error \\")
lines.append(r"\hline")

for _, row in article_table.iterrows():
    choice = str(row["choice"]).replace("_", r"\_")
    parameters = row["parameters"]
    line = (
        f"{choice} & {parameters} & "
        f"{fmt(row['mean_error_Fourier'])} & "
        f"{fmt(row['mean_error_Couette'])} & "
        f"{fmt(row['mean_error_all'])} & "
        f"{fmt(row['max_error_all'])} \\\\"
    )
    lines.append(line)

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
lines.append("")

tex = "\n".join(lines)
(OUT_TABLE / "table_default_vs_fitted.tex").write_text(tex)

print("Wrote:")
print("  results/data/exp13_default_vs_fitted_summary.csv")
print("  results/tables/table_default_vs_fitted.csv")
print("  results/tables/table_default_vs_fitted.tex")
print()
print(article_table.to_string(index=False))
