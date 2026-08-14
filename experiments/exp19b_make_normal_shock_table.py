from pathlib import Path
import pandas as pd

data = Path("results/data/exp19_dvm_bgk_normal_shock_metrics.csv")
outdir = Path("results/tables")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data)

cols = [
    "Mach", "Kn", "max_K_local", "max_Wf_default", "max_Wf_fitted",
    "shock_thickness", "final_residual", "residual_tol", "converged"
]
df[cols].to_csv(outdir / "table_exp19_normal_shock_metrics.csv", index=False)

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Reduced DVM/BGK normal-shock diagnostic. The test uses a one-dimensional BGK kinetic model to examine activation of the local rarefaction indicator inside shock layers. It is a kinetic diagnostic, not a DSMC validation of monatomic-gas shock structure.}",
    r"\label{tab:exp19_normal_shock}",
    r"\begin{tabular}{cccccccc}",
    r"\toprule",
    r"$M_1$ & Kn & $\max K_{\rm local}$ & $\max W_f$ default & $\max W_f$ fitted & Thickness & Residual & Conv. \\",
    r"\midrule",
]

for _, r in df.iterrows():
    conv = "yes" if bool(r["converged"]) else "no"
    lines.append(
        f"{r['Mach']:.1f} & {r['Kn']:.2f} & {r['max_K_local']:.4f} & "
        f"{r['max_Wf_default']:.4f} & {r['max_Wf_fitted']:.4f} & "
        f"{r['shock_thickness']:.4f} & {r['final_residual']:.2e} & {conv} \\\\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

(outdir / "table_exp19_normal_shock_metrics.tex").write_text("\n".join(lines) + "\n")

print("wrote results/tables/table_exp19_normal_shock_metrics.csv")
print("wrote results/tables/table_exp19_normal_shock_metrics.tex")
print(df[cols].to_string(index=False))
