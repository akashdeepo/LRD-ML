"""
MODULE 12: Incremental tests against HAR-X and Sharpe-ratio inference
=====================================================================
Pre-registered on 2026-09-14 (docs/EXPERIMENTS.md, run-03). Two parts.

Part A - forecast accuracy beyond HAR-X (Model A1), per horizon:
    * pooled MSE and % change relative to HAR and to HAR-X
    * panel-aware HLN-corrected Diebold-Mariano vs HAR-X (module 6), two-sided
    * Clark-West (2007) MSPE-adjusted test vs HAR-X, one-sided, for the
      nested specifications (A1cs, A1sec, A1mod, C all nest A1). Computed on
      the cross-sectional mean adjusted loss differential per date with the
      same Newey-West bandwidth ceil(h/5)-1 as the DM test.
    * Holm-adjusted CW p-values across the nine pre-registered tests
      (three new specifications x three horizons); C reported separately.
    * regime splits (low-VIX, high-VIX, COVID) descriptively, with DM t.
  Outputs results/tables/table11_harx_tests.tex, results/intermediate/harx_tests.csv,
  results/intermediate/harx_tests_regimes.csv.

Part B - volatility-managed portfolio Sharpe differences (h = 5):
    * C-managed vs HAR-X-managed, C-managed vs unmanaged, HAR-X vs unmanaged
    * Ledoit-Wolf (2008) HAC delta-method test on the difference of Sharpe
      ratios and a studentized circular block bootstrap p-value
    * full sample and the three regimes (COVID has 43 weekly observations;
      reported, not relied on)
  Outputs results/tables/table12_sharpe_tests.tex, results/intermediate/sharpe_tests.csv.

Run from the repo root:  python -m modules.module12_incremental_tests
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modules.forecast_io import HORIZONS, load_bundle
from modules.module6_forecast_eval import (
    SAMPLE_STRIDE, diebold_mariano, load_forecast_panels, qlike_loss, regime_masks,
    squared_loss,
)
from modules import module11_economic as m11

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
TABLES = BASE / "results" / "tables"
INTERM = BASE / "results" / "intermediate"

BASELINE = "A1"
PREREG_SPECS = ["A1cs", "A1sec", "A1mod"]          # the nine Holm-adjusted tests
OTHER_SPECS = ["C", "A2", "A3", "A4", "A5",
               "D_lasso", "D_ridge", "D_en", "D_rf", "D_gbm"]


# ------------------------------------------------------------------ helpers
def _panels(model: str, h: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    yh = FCST / f"{model}_h{h:02d}_yhat.csv"
    y = FCST / f"{model}_h{h:02d}_y.csv"
    if not yh.exists():
        return None
    return (pd.read_csv(yh, index_col=0, parse_dates=True),
            pd.read_csv(y, index_col=0, parse_dates=True))


def _hac_mean_test(d: np.ndarray, h: int) -> tuple[float, float, float, int]:
    """t-statistic for mean(d) = 0 with Newey-West (Bartlett) HAC variance at
    bandwidth ceil(h/stride) - 1. Returns (mean, se, t, T)."""
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 5:
        return np.nan, np.nan, np.nan, T
    m = d.mean()
    c = d - m
    bw = max(int(np.ceil(h / SAMPLE_STRIDE)), 1) - 1
    v = float((c ** 2).mean())
    for k in range(1, bw + 1):
        v += 2.0 * (1.0 - k / (bw + 1)) * float((c[k:] * c[:-k]).mean())
    se = float(np.sqrt(max(v, 1e-300) / T))
    return float(m), se, float(m / se), T


def clark_west(yhat_small: pd.DataFrame, yhat_big: pd.DataFrame, y: pd.DataFrame,
               h: int, mask=None) -> dict:
    """Clark-West (2007) test that the nested (small) model's population MSPE
    equals the larger model's. Adjusted differential
        f_t = e_small^2 - [ e_big^2 - (yhat_small - yhat_big)^2 ],
    averaged across stocks per date, then a HAC t-test; one-sided p from the
    standard normal (Clark-West's recommendation)."""
    idx = yhat_small.index.intersection(yhat_big.index).intersection(y.index)
    cols = yhat_small.columns.intersection(yhat_big.columns).intersection(y.columns)
    ys, yb, yy = (df.loc[idx, cols] for df in (yhat_small, yhat_big, y))
    f = (yy - ys) ** 2 - ((yy - yb) ** 2 - (ys - yb) ** 2)
    if mask is not None:
        f = f.loc[mask]
    d = f.mean(axis=1).values
    m, se, t, T = _hac_mean_test(d, h)
    p = float(1 - stats.norm.cdf(t)) if np.isfinite(t) else np.nan
    return {"cw_mean": m, "cw_t": t, "cw_p_one_sided": p, "T": T}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()


# ------------------------------------------------------------------ part A
def part_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = load_bundle()
    rows, rows_reg = [], []
    base = {h: _panels(BASELINE, h) for h in HORIZONS}
    har = {h: _panels("A", h) for h in HORIZONS}
    for spec in PREREG_SPECS + OTHER_SPECS:
        for h in HORIZONS:
            got = _panels(spec, h)
            if got is None or base[h] is None:
                continue
            yh, y = got
            yh_b, y_b = base[h]
            yh_a, y_a = har[h]
            L, Lb, La = squared_loss(yh, y), squared_loss(yh_b, y_b), squared_loss(yh_a, y_a)
            Q, Qb = qlike_loss(yh, y), qlike_loss(yh_b, y_b)
            mse, mse_b, mse_a = (float(np.nanmean(x.values)) for x in (L, Lb, La))
            _, t_dm, p_dm, T = diebold_mariano(Lb, L, h=h)
            _, t_dm_q, p_dm_q, _ = diebold_mariano(Qb, Q, h=h)
            _, t_dm_a, p_dm_a, _ = diebold_mariano(La, L, h=h)
            cw = clark_west(yh_b, yh, y, h)
            rows.append({
                "spec": spec, "h": h, "T_dates": T,
                "mse": mse, "mse_harx": mse_b, "mse_har": mse_a,
                "gain_vs_har_pct": 100 * (1 - mse / mse_a),
                "gain_vs_harx_pct": 100 * (1 - mse / mse_b),
                "gain_vs_harx_pp": 100 * (1 - mse / mse_a) - 100 * (1 - mse_b / mse_a),
                "dm_t_vs_har": t_dm_a, "dm_p_vs_har": p_dm_a,
                "dm_t_vs_harx": t_dm, "dm_p_vs_harx": p_dm,
                "dm_t_vs_harx_qlike": t_dm_q, "dm_p_vs_harx_qlike": p_dm_q,
                "cw_t_vs_harx": cw["cw_t"], "cw_p_vs_harx": cw["cw_p_one_sided"],
            })
            # regimes (descriptive)
            idx = L.index
            for name, m in regime_masks(idx, bundle.market).items():
                if m.sum() < 20:
                    continue
                mse_r = float(np.nanmean(L.loc[m].values))
                mse_br = float(np.nanmean(Lb.loc[m].values))
                mse_ar = float(np.nanmean(La.loc[m].values))
                _, t_r, p_r, T_r = diebold_mariano(Lb.loc[m], L.loc[m], h=h)
                cw_r = clark_west(yh_b, yh, y, h, mask=m)
                rows_reg.append({
                    "spec": spec, "h": h, "regime": name, "T_dates": T_r,
                    "gain_vs_har_pct": 100 * (1 - mse_r / mse_ar),
                    "gain_vs_harx_pp": 100 * (1 - mse_r / mse_ar) - 100 * (1 - mse_br / mse_ar),
                    "dm_t_vs_harx": t_r, "dm_p_vs_harx": p_r,
                    "cw_t_vs_harx": cw_r["cw_t"], "cw_p_vs_harx": cw_r["cw_p_one_sided"],
                })
    df = pd.DataFrame(rows)
    # Holm across the nine pre-registered CW tests; C separately across its three
    pre = df["spec"].isin(PREREG_SPECS)
    df["cw_p_holm"] = np.nan
    df.loc[pre, "cw_p_holm"] = holm(df.loc[pre, "cw_p_vs_harx"].tolist())
    isC = df["spec"] == "C"
    df.loc[isC, "cw_p_holm"] = holm(df.loc[isC, "cw_p_vs_harx"].tolist())
    df["adds_beyond_harx"] = (df["cw_p_holm"] < 0.05) & (df["gain_vs_harx_pct"] > 0)
    df.to_csv(INTERM / "harx_tests.csv", index=False)
    pd.DataFrame(rows_reg).to_csv(INTERM / "harx_tests_regimes.csv", index=False)
    _write_table11(df)
    return df, pd.DataFrame(rows_reg)


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def _write_table11(df: pd.DataFrame) -> None:
    names = {"A1cs": "HAR-X + cross-sectional $\\bar d_t$, $\\sigma_d^t$",
             "A1sec": "HAR-X + sector-mean $\\bar d_{s(i),t}$",
             "A1mod": "HAR-X + $\\bar d_t$ $\\times$ HAR components",
             "C": "Model $C$ (full union)",
             "D_lasso": "Model $D$, Lasso", "D_ridge": "Model $D$, Ridge",
             "D_en": "Model $D$, Elastic Net", "D_rf": "Model $D$, Random Forest",
             "D_gbm": "Model $D$, Gradient Boosting"}
    order = ["A1cs", "A1sec", "A1mod", "C", "D_lasso", "D_ridge", "D_en", "D_rf", "D_gbm"]
    with open(TABLES / "table11_harx_tests.tex", "w", encoding="utf-8") as f:
        f.write("% Table 11: incremental tests vs HAR-X (point-in-time forecasts, 2026-09-14)\n\n")
        f.write("\\begin{table}[htbp]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\centering\n")
        f.write("\\caption{Incremental Forecast Accuracy Relative to HAR-X: Point-in-Time Out-of-Sample Tests}\n")
        f.write("\\label{tab:harx_tests}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
        f.write("Specification & $h$ & \\%$\\Delta$MSE vs HAR-X & HLN DM-$t$ & CW-$t$ & CW $p$ (Holm) & QLIKE DM-$t$ \\\\\n\\midrule\n")
        for spec in order:
            sub = df[df["spec"] == spec].sort_values("h")
            if sub.empty:
                continue
            for i, r in enumerate(sub.itertuples()):
                lab = names.get(spec, spec) if i == 0 else ""
                holm_p = f"{r.cw_p_holm:.3f}" if np.isfinite(r.cw_p_holm) else "--"
                f.write(f"{lab} & {r.h} & {r.gain_vs_harx_pct:+.2f}\\% & "
                        f"${r.dm_t_vs_harx:+.2f}${_stars(r.dm_p_vs_harx)} & "
                        f"${r.cw_t_vs_harx:+.2f}${_stars(r.cw_p_vs_harx)} & {holm_p} & "
                        f"${r.dm_t_vs_harx_qlike:+.2f}${_stars(r.dm_p_vs_harx_qlike)} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: All forecasts are point-in-time (training rows whose target window "
                "extends past the origin are excluded). \\%$\\Delta$MSE is the change in pooled MSE "
                "on $\\log RV^{PK}$ relative to HAR-X (Model $A_1$); positive favours the "
                "specification. HLN DM-$t$: panel-aware Harvey--Leybourne--Newbold-corrected "
                "Diebold--Mariano statistic against HAR-X (two-sided). CW-$t$: Clark--West (2007) "
                "MSPE-adjusted statistic for the nested comparison with HAR-X (one-sided, standard "
                "normal). Both use the cross-sectional mean loss differential per date with a "
                "Newey--West bandwidth $\\lceil h/5\\rceil-1$. CW $p$ (Holm) adjusts the three "
                "pre-registered specifications' nine tests jointly and Model $C$'s three tests "
                "jointly; other rows are unadjusted (--). Significance of raw statistics: "
                "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.\n")
        f.write("\\end{tablenotes}\n\\end{adjustwidth}\n\\end{table}\n")
    print(f"Saved {TABLES / 'table11_harx_tests.tex'}")


# ------------------------------------------------------------------ part B
def _sharpe_diff_hac(r1: np.ndarray, r2: np.ndarray, bw: int) -> tuple[float, float, float]:
    """Ledoit-Wolf (2008) delta-method HAC test of SR1 - SR2 = 0.
    Returns (delta, se, t)."""
    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    delta = mu1 / s1 - mu2 / s2
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -mu1 / (2 * s1 ** 3), mu2 / (2 * s2 ** 3)])
    V = np.column_stack([r1, r2, r1 ** 2, r2 ** 2])
    V = V - V.mean(axis=0)
    T = len(V)
    Psi = V.T @ V / T
    for k in range(1, bw + 1):
        G = V[k:].T @ V[:-k] / T
        Psi += (1 - k / (bw + 1)) * (G + G.T)
    se = float(np.sqrt(grad @ Psi @ grad / T))
    return float(delta), se, float(delta / se)


def _circular_block_bootstrap_p(r1: np.ndarray, r2: np.ndarray, bw: int,
                                block: int, B: int, seed: int = 0) -> float:
    """Studentized circular block bootstrap p-value for the Sharpe difference
    (Ledoit-Wolf 2008, Section 3.2): centre the bootstrap statistic at the
    sample difference and compare |t*| with |t|."""
    rng = np.random.default_rng(seed)
    T = len(r1)
    d_hat, _, t_hat = _sharpe_diff_hac(r1, r2, bw)
    n_blocks = int(np.ceil(T / block))
    count = 0
    for _ in range(B):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        d_b, se_b, _ = _sharpe_diff_hac(r1[idx], r2[idx], bw)
        if se_b > 0 and abs((d_b - d_hat) / se_b) >= abs(t_hat):
            count += 1
    return (count + 1) / (B + 1)


def part_b(B: int = 2000, block: int = 5) -> pd.DataFrame:
    bundle = load_bundle()
    sd = bundle.sample_dates
    ret_h = m11._five_day_log_returns(bundle, sd)
    port = {"Unmanaged": ret_h.mean(axis=1, skipna=True)}
    for model in ["A1", "C"]:
        yhat = m11._load_model_yhat(model)
        common = yhat.index.intersection(ret_h.index)
        keep = yhat.loc[common].notna().any(axis=1)
        common = common[keep]
        port[model], _ = m11._vol_managed_returns(yhat.loc[common], ret_h.loc[common])
    common = port["A1"].dropna().index.intersection(port["C"].dropna().index)
    P = pd.DataFrame({k: v.reindex(common) for k, v in port.items()}).dropna()
    masks = m11.regime_masks_5d(P.index, bundle.market)
    ann = np.sqrt(m11.ANN_PER_PERIOD) if hasattr(m11, "ANN_PER_PERIOD") else np.sqrt(252 / 5)
    rows = []
    for regime, m in masks.items():
        sub = P.loc[m]
        if len(sub) < 20:
            continue
        for a, b in [("C", "A1"), ("C", "Unmanaged"), ("A1", "Unmanaged")]:
            r1, r2 = sub[a].values, sub[b].values
            d, se, t = _sharpe_diff_hac(r1, r2, bw=0)          # non-overlapping weekly returns
            p_hac = float(2 * (1 - stats.norm.cdf(abs(t))))
            p_boot = _circular_block_bootstrap_p(r1, r2, bw=0, block=block, B=B)
            rows.append({"regime": regime, "pair": f"{a} vs {b}", "T_weeks": len(sub),
                         "sharpe_1_ann": float(r1.mean() / r1.std(ddof=0) * ann),
                         "sharpe_2_ann": float(r2.mean() / r2.std(ddof=0) * ann),
                         "delta_sharpe_ann": d * ann, "hac_t": t, "hac_p": p_hac,
                         "boot_p": p_boot, "block": block, "B": B})
            print(f"  {regime:14s} {a:>3s} vs {b:9s}  dSR={d*ann:+.3f}  t={t:+.2f}  p_HAC={p_hac:.3f}  p_boot={p_boot:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "sharpe_tests.csv", index=False)
    with open(TABLES / "table12_sharpe_tests.tex", "w", encoding="utf-8") as f:
        f.write("% Table 12: Sharpe-ratio difference tests (Ledoit-Wolf 2008)\n\n")
        f.write("\\begin{table}[htbp]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n\\centering\n")
        f.write("\\caption{Volatility-Managed Portfolios: Tests of Sharpe-Ratio Differences}\n")
        f.write("\\label{tab:sharpe_tests}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llcccccc}\n\\toprule\n")
        f.write("Regime & Comparison & Weeks & Sharpe (1) & Sharpe (2) & $\\Delta$Sharpe & HAC $t$ & Bootstrap $p$ \\\\\n\\midrule\n")
        for r in df.itertuples():
            f.write(f"{r.regime} & {r.pair.replace('A1', 'HAR-X').replace('C', 'Model C', 1)} & {r.T_weeks} & "
                    f"{r.sharpe_1_ann:.2f} & {r.sharpe_2_ann:.2f} & {r.delta_sharpe_ann:+.2f} & "
                    f"${r.hac_t:+.2f}${_stars(r.hac_p)} & {r.boot_p:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Weekly (five-trading-day, non-overlapping) portfolio returns on the "
                "forecast sample stride; Sharpe ratios annualised by $\\sqrt{252/5}$. HAC $t$ is the "
                "Ledoit--Wolf (2008) delta-method statistic for the difference in Sharpe ratios; "
                f"bootstrap $p$ is from a studentised circular block bootstrap (block length {block}, "
                f"{B} resamples). Significance of the HAC statistic: $^{{*}}$ $p<0.10$, $^{{**}}$ $p<0.05$, "
                "$^{***}$ $p<0.01$. Regime rows use the same VIX quartiles and COVID window as the regime table.\n")
        f.write("\\end{tablenotes}\n\\end{adjustwidth}\n\\end{table}\n")
    print(f"Saved {TABLES / 'table12_sharpe_tests.tex'}")
    return df


def main() -> None:
    print("=" * 70)
    print("   MODULE 12: INCREMENTAL TESTS VS HAR-X + SHARPE INFERENCE")
    print("=" * 70)
    df, reg = part_a()
    cols = ["spec", "h", "gain_vs_har_pct", "gain_vs_harx_pct", "dm_t_vs_harx", "dm_p_vs_harx",
            "cw_t_vs_harx", "cw_p_vs_harx", "cw_p_holm", "dm_t_vs_harx_qlike", "adds_beyond_harx"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))
        print("\nRegimes (descriptive):")
        print(reg[reg["spec"].isin(PREREG_SPECS + ["C"])].to_string(index=False))
    print("\nPart B: Sharpe-ratio differences")
    part_b()


if __name__ == "__main__":
    main()
