from pathlib import Path
import pandas as pd

data = Path("results/data/exp17_flux_moment_validation_grouped.csv")
outdir = Path("results/tables")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data)

bench_order = {"Fourier": 0, "Couette": 1}
moment_order = {"heat_flux": 0, "velocity_profile": 1, "shear_stress": 2}
model_order = {"nsf": 0, "hard": 1, "algebraic": 2, "log": 3, "jump": 4, "slip": 5}

df["bo"] = df["benchmark"].map(bench_order).fillna(9)
df["mo"] = df["moment"].map(moment_order).fillna(9)
df["lo"] = df["model"].map(model_order).fillna(9)
df = df.sort_values(["bo", "mo", "lo"])

cols = ["benchmark", "moment", "model", "mean_error", "max_error", "n"]
df[cols].to_csv(outdir / "table_flux_moment_validation.csv", index=False)

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{DVM/BGK non-equilibrium flux and moment validation. Errors are relative to DVM/BGK reference values. Rarefaction branches improve heat-flux and shear-stress predictions relative to NSF, while the log-Gaussian limiter is not always the best moment closure.}",
    r"\label{tab:flux_moment_validation}",
    r"\begin{tabular}{lllccc}",
    r"\toprule",
    r"Benchmark & Moment & Model & Mean error & Max error & Cases \\",
    r"\midrule",
]

for _, r in df.iterrows():
    moment = str(r["moment"]).replace("_", r"\_")
    lines.append(
        f"{r['benchmark']} & {moment} & {r['model']} & "
        f"{r['mean_error']:.4e} & {r['max_error']:.4e} & {int(r['n'])} " + r"\\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

(outdir / "table_flux_moment_validation.tex").write_text("\n".join(lines) + "\n")

print("wrote results/tables/table_flux_moment_validation.csv")
print("wrote results/tables/table_flux_moment_validation.tex")
print(df[cols].to_string(index=False))
