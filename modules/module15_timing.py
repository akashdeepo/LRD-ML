"""
MODULE 15: How Much of HAR-X's Gain Is Information Timing?
=========================================================
HAR inputs must use the same information set as the implied-volatility
regressors. Here HAR and HAR-X are re-estimated with the HAR terms built from
RV through day t-1 and the return term r_{t-1}, while VIX and MOVE stay at the
close of day t (the pre-audit convention; audit I1). Everything else matches
module 4: expanding per-stock OLS, point-in-time embargo, common evaluation
date, common cells. Compared with the aligned A and A1 forecasts (ledger
run-16).

Outputs:
    results/intermediate/timing_alignment.csv
    results/tables/table15_timing.tex
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules.forecast_io import (
    HORIZONS, aligned_xy, build_targets, load_bundle, stock_matrix,
)
from modules.module3_feature_engineering import build_har
from modules.module4_benchmarks import expanding_forecast
from modules.module5_ml_models import INIT_TRAIN_FRAC
from modules.module6_forecast_eval import common_cells, diebold_mariano, squared_loss

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
TABLES = BASE / "results" / "tables"


def stale_inputs(bundle) -> dict[str, pd.DataFrame]:
    """HAR terms from RV through t-1 and r_{t-1}, on the weekly sample grid."""
    har = build_har(bundle.rv.shift(1))
    out = {f"{k}_log": np.log(v.where(v > 0)).reindex(bundle.sample_dates)
           for k, v in har.items()}
    r = bundle.returns.shift(1).reindex(bundle.sample_dates)
    out["ret_lag1"], out["ret_lag1_abs"] = r, r.abs()
    return out


def _one_stock(X: pd.DataFrame, y: pd.Series, init_date, h: int, full_index) -> pd.Series:
    return expanding_forecast(X, y, init_date, h, full_index)


def stale_forecasts(bundle, model: str, h: int, stale: dict, targets) -> tuple[pd.DataFrame, pd.DataFrame]:
    init_date = bundle.sample_dates[int(len(bundle.sample_dates) * INIT_TRAIN_FRAC)]
    jobs = []
    for t in bundle.panel.kept:
        sm = stock_matrix(bundle, t, model, targets)
        for c, panel in stale.items():
            sm.X[c] = panel[t]
        X, y = aligned_xy(sm, h)
        if len(X) >= 60:
            jobs.append((t, X, y))
    fits = Parallel(n_jobs=12)(delayed(_one_stock)(X, y, init_date, h, bundle.rv.index)
                               for _, X, y in jobs)
    yhat = pd.DataFrame(index=bundle.sample_dates, columns=bundle.panel.kept, dtype=float)
    yy = yhat.copy()
    for (t, X, y), f in zip(jobs, fits):
        yhat.loc[f.index, t] = f.values
        yy.loc[y.index, t] = y.values
    return yhat, yy


def _load(model: str, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rd = lambda s: pd.read_csv(FCST / f"{model}_h{h:02d}_{s}.csv", index_col=0, parse_dates=True)
    return rd("yhat"), rd("y")


def main() -> None:
    print("=" * 70)
    print("   MODULE 15: TIMING ALIGNMENT")
    print("=" * 70)
    bundle = load_bundle()
    targets = build_targets(bundle)
    stale = stale_inputs(bundle)
    rows = []
    for h in HORIZONS:
        loss = {}
        for model in ("A", "A1"):
            yh, y = _load(model, h)
            loss[(model, "aligned")] = squared_loss(yh, y)
            yh_s, y_s = stale_forecasts(bundle, model, h, stale, targets)
            loss[(model, "stale")] = squared_loss(yh_s, y_s)
        # one common set of cells across all four forecast panels
        mask = None
        for L in loss.values():
            m = L.notna()
            mask = m if mask is None else (mask & m.reindex_like(mask).fillna(False))
        loss = {k: v.reindex_like(mask).where(mask) for k, v in loss.items()}
        mse = {k: float(np.nanmean(v.values)) for k, v in loss.items()}
        for align in ("aligned", "stale"):
            _, t_dm, p_dm, T = diebold_mariano(loss[("A", align)], loss[("A1", align)], h=h)
            rows.append({"h": h, "alignment": align, "T_dates": T,
                         "mse_har": mse[("A", align)], "mse_harx": mse[("A1", align)],
                         "harx_gain_pct": 100 * (1 - mse[("A1", align)] / mse[("A", align)]),
                         "dm_t": t_dm, "dm_p": p_dm})
        _, t_h, p_h, _ = diebold_mariano(loss[("A", "stale")], loss[("A", "aligned")], h=h)
        rows.append({"h": h, "alignment": "har_aligned_vs_stale", "T_dates": T,
                     "mse_har": mse[("A", "aligned")], "mse_harx": np.nan,
                     "harx_gain_pct": 100 * (1 - mse[("A", "aligned")] / mse[("A", "stale")]),
                     "dm_t": t_h, "dm_p": p_h})
        print(pd.DataFrame(rows[-3:]).round(4).to_string(index=False))
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "timing_alignment.csv", index=False)
    write_table(df, TABLES / "table15_timing.tex")


def write_table(df: pd.DataFrame, fp: Path) -> None:
    with open(fp, "w", encoding="utf-8") as f:
        f.write("% Table 15: information timing (module 15)\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{HAR-X's gain over HAR under aligned and one-day-stale HAR inputs}\n")
        f.write("\\label{tab:timing}\n\\small\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("& $h$ & MSE HAR & MSE HAR-X & HAR-X gain (DM-$t$) \\\\\n\\midrule\n")
        for align, lab in (("stale", "HAR inputs through $t-1$"), ("aligned", "HAR inputs through $t$")):
            sub = df[df["alignment"] == align]
            for i, r in enumerate(sub.itertuples()):
                f.write(f"{lab if i == 0 else ''} & {r.h} & {r.mse_har:.4f} & {r.mse_harx:.4f} & "
                        f"{r.harx_gain_pct:+.2f}\\% ({r.dm_t:+.2f}) \\\\\n")
            f.write("\\midrule\n")
        sub = df[df["alignment"] == "har_aligned_vs_stale"]
        for i, r in enumerate(sub.itertuples()):
            f.write(f"{'HAR: aligned vs stale' if i == 0 else ''} & {r.h} & & & "
                    f"{r.harx_gain_pct:+.2f}\\% ({r.dm_t:+.2f}) \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: VIX and MOVE are closes of the origin day $t$ in every row. "
                "\\emph{Stale}: HAR terms from $RV$ through $t-1$ and the return term $r_{t-1}$. "
                "\\emph{Aligned}: through $t$. Pooled MSE on $\\log RV^{PK}$ over common "
                "(date, stock) cells of all four forecast panels; gains and panel HLN DM-$t$ "
                "statistics for HAR-X against HAR, and for aligned against stale HAR.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"  saved {fp.name}")


if __name__ == "__main__":
    main()
