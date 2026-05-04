"""Regenerate table9_robustness.tex from the cached table9_raw.csv with the
new resizebox+condensed-inference formatting. Avoids the 30-minute window
rerun in module9_robustness.main()."""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
TABLES_DIR = BASE / "results" / "tables"

df = pd.read_csv(INTERM / "table9_raw.csv")

LABELS = {
    "headline": "Headline (Model C, full panel)",
    "inference_plain": "Inference: plain DM (illustrative)",
    "benchmark_garch11": "Benchmark: GARCH(1,1) on returns",
    "liquidity_high_illiq": "Liquidity: low-liquidity half",
    "liquidity_low_illiq": "Liquidity: high-liquidity half",
    "estimator_LW": r"Estimator: $\hat d_{LW}$ replaces $\hat d_{GPH}$",
    "window_500": "Rolling window: 500 days",
    "window_1000": "Rolling window: 1000 days",
    "target_sqret": "Target: log mean future squared returns",
    "regime_high_vix": "Regime: high VIX (Q4)",
    "regime_low_vix": "Regime: low VIX (Q1)",
    "regime_covid": "Regime: COVID 2020",
    "regime_gfc": "Regime: GFC 2008--2009",
}

fp = TABLES_DIR / "table9_robustness.tex"
with open(fp, "w") as f:
    f.write("% Table 9: Robustness summary -- Model C vs Model A at h=5\n\n")
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Robustness Summary: Model C vs Model A, $h=5$}\n")
    f.write("\\label{tab:robustness}\n\\small\n")
    f.write("\\resizebox{\\textwidth}{!}{%\n")
    f.write("\\begin{tabular}{lccc}\n\\toprule\n")
    f.write("Variant & MSE & \\%$\\Delta$ vs A & HLN DM-$t$ \\\\\n")
    f.write("\\midrule\n")
    for _, r in df.iterrows():
        label = LABELS.get(r["variant"], r["variant"])
        mse_val = r.get("MSE", np.nan)
        mse = "--" if pd.isna(mse_val) else f"{mse_val:.4f}"
        imp_pct_val = r.get("imp_pct", np.nan)
        imp = "--" if pd.isna(imp_pct_val) else f"{imp_pct_val:+.2f}\\%"
        t_val = r.get("HLN_DM_t", np.nan)
        t = "--" if pd.isna(t_val) else f"${t_val:+.2f}$"
        if r["variant"] == "inference_plain":
            t = (f"${r.get('HLN_DM_t', np.nan):+.2f}$ "
                 f"(plain $+{r.get('plain_DM_t', np.nan):.2f}$)")
        f.write(f"{label} & {mse} & {imp} & {t} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}%\n}\n")
    f.write("\\begin{tablenotes}\\small\n")
    f.write(
        "\\item Notes: Robustness checks for Model C against Model A at "
        "$h=5$ on $\\log RV^{PK}$. Liquidity halves are formed from the "
        "static median of mean inverse dollar volume across the sample. "
        "$\\hat d_{LW}$ replaces $\\hat d_{GPH}$ in the entire feature "
        "block (including memory dynamics and interactions). Window "
        "variants refit the rolling LRD estimation at the alternative "
        "window before regenerating the full feature stack. The target "
        "variant replaces $\\log RV^{PK}$ with log mean future squared "
        "returns (Models A and C both refit). The GARCH(1,1) row fits a "
        "constant-mean GARCH(1,1) on log returns with refit every 20 "
        "sample steps and reports the resulting log mean conditional "
        "variance forecast against the same Parkinson target; its "
        "forecast scale differs from the Parkinson RV target (returns "
        "variance vs.\\ range-based variance proxy), so the negative "
        "$\\%\\Delta$ vs A reflects both this level mismatch and the "
        "well-known limitation of returns-only GARCH for forecasting "
        "realised variance. Regime rows come from the Table 7 split at "
        "$h=5$. The plain-DM stat is the unscaled iid pooled-cell "
        "statistic; the HLN stat is the panel-aware "
        "Harvey-Leybourne-Newbold finite-sample-corrected version.\n"
    )
    f.write("\\end{tablenotes}\n\\end{table}\n")
print(f"Saved {fp}")
