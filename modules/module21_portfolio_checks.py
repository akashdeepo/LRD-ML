"""
MODULE 21: Portfolio Implementation Checks (run-25)
===================================================
Variants of the module-11 volatility-managed portfolios requested in the
co-author review:

  * execution one trading day after the signal (weights from the close of t,
    positions held over t+2..t+6), since the VIX closes after the equity
    market and the day's range is known only at the close;
  * a cap of 2 on the per-stock position multiplier;
  * both together;

for each: Sharpe ratios at 0 and 10 bp, mean, volatility, maximum drawdown,
turnover, leverage distribution, and the Ledoit-Wolf and bootstrap tests of
C-managed vs HAR-X-managed and vs equal weight (module 12 helpers).

Output: results/intermediate/portfolio_variants.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modules.forecast_io import load_bundle
from modules import module11_economic as m11
from modules.module12_incremental_tests import _circular_block_bootstrap_p, _sharpe_diff_hac

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"

VARIANTS = {"baseline (same-close execution, no cap)": dict(exec_lag=0, cap=None),
            "next-day execution": dict(exec_lag=1, cap=None),
            "cap 2": dict(exec_lag=0, cap=2.0),
            "next-day execution, cap 2": dict(exec_lag=1, cap=2.0)}
PAIRS = [("C-managed", "HAR-X-managed"), ("C-managed", m11.UNMANAGED), ("HAR-X-managed", m11.UNMANAGED)]


def main() -> None:
    pd.set_option("display.width", 220)
    print("=" * 70); print("   MODULE 21: PORTFOLIO CHECKS"); print("=" * 70)
    bundle = load_bundle()
    rows, tests = [], []
    for name, kw in VARIANTS.items():
        P, diag = m11.build_portfolios(bundle, **kw)
        rf = diag["rf"]
        for lab in m11.ORDER:
            w = diag["weights"][lab]
            to = m11.turnover(w, diag["R"], rf).reindex(P.index).fillna(0.0)
            st = m11.stats(P[lab], rf)
            gross = w.sum(axis=1) / w.notna().sum(axis=1)
            wv = w.stack()
            rows.append({"variant": name, "portfolio": lab, "weeks": len(P), **st,
                         "sharpe_10bp": m11.stats(P[lab] - 10 / 1e4 * to, rf)["sharpe"],
                         "turnover_ann": float(to.mean() * m11.ANN_PER_PERIOD),
                         "w_p99": float(wv.quantile(0.99)), "w_max": float(wv.max()),
                         "gross_mean": float(gross.mean()), "gross_max": float(gross.max())})
        bw = 6
        for a, b in PAIRS:
            r1, r2 = P[a].values, P[b].values
            ok = ~(np.isnan(r1) | np.isnan(r2))
            d, _, hac_t = _sharpe_diff_hac(r1[ok], r2[ok], bw)
            hac_p = float(2 * (1 - stats.norm.cdf(abs(hac_t))))
            boot_p = _circular_block_bootstrap_p(r1[ok], r2[ok], bw, block=5, B=2000)
            tests.append({"variant": name, "pair": f"{a} vs {b}", "delta_sharpe_ann": d,
                          "hac_t": hac_t, "hac_p": hac_p, "boot_p": boot_p})
        print(f"  {name}: {len(P)} weeks")
    df, tdf = pd.DataFrame(rows), pd.DataFrame(tests)
    df.to_csv(INTERM / "portfolio_variants.csv", index=False)
    tdf.to_csv(INTERM / "portfolio_variants_tests.csv", index=False)
    print(df.pivot_table(index="portfolio", columns="variant", values="sharpe").round(3).to_string())
    print(df.pivot_table(index="portfolio", columns="variant", values="max_dd").round(3).to_string())
    print(tdf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
