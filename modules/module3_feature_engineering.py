"""
MODULE 3: Persistence-Based Feature Engineering (v2)
====================================================
Builds the persistence feature vector Z_t per Rachev's framework Sec. 6:

  Z_t = ( d_GPH_t,  Δd_GPH_t,  Vol(d)_t,  Trend(d)_t,
          d_LW_t,   ΔLW_t,
          H_t,      ΔH_t,
          d̄_t,      σ_d^t,    skew_d^t,   kurt_d^t,   range_d^t,
          d̄_{s,t}  (per-sector mean),
          1{d_t > τ}_τ  (threshold indicators),
          d_t * VIX_t,  d_t * MOVE_t,  d_t * (1/Liq_t)  (interactions),
          HAR components: RV_d, RV_w, RV_m on Parkinson RV )

The module consumes Phase 2 outputs (rolling_d_gph.csv, rolling_d_lw.csv,
rolling_hurst.csv) so that estimation does not have to be repeated.

Outputs (under results/intermediate/features/):
  feat_d_gph.csv, feat_d_lw.csv, feat_h.csv, feat_delta_d_gph.csv, ...
  feat_har_d.csv, feat_har_w.csv, feat_har_m.csv,
  feat_cross_section.csv  (market-level: mean/std/skew/kurt/range_d, VIX, MOVE, ...)
  feat_sector_mean_d.csv  (T_sample x N: each stock gets its sector's mean d̂)
  feat_thresholds.csv     (T_sample x (N x K) flags above τ_k)
  feat_interactions.csv   (T_sample x (N x I) interaction terms)
  market_axes.csv         (the broadcast market vector aligned to stride)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from modules.io_v2 import build_clean_panel, sector_map

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
TABLES_DIR = BASE / "results" / "tables"
INTERMEDIATE_DIR = BASE / "results" / "intermediate"
FEAT_DIR = INTERMEDIATE_DIR / "features"

HAR_DAILY = 1
HAR_WEEKLY = 5
HAR_MONTHLY = 22

DYNAMICS_LAG = 22       # days for delta, vol, trend (in rolling-cadence steps once we re-stride)
THRESHOLDS = (0.10, 0.20, 0.30, 0.40)
LIQUIDITY_FLOOR = 1e3   # avoid division blowup on illiquid days


def load_phase2() -> dict[str, pd.DataFrame]:
    files = {
        "d_gph": INTERMEDIATE_DIR / "rolling_d_gph.csv",
        "d_lw": INTERMEDIATE_DIR / "rolling_d_lw.csv",
        "h": INTERMEDIATE_DIR / "rolling_hurst.csv",
    }
    out = {}
    for k, fp in files.items():
        out[k] = pd.read_csv(fp, index_col=0, parse_dates=True)
        print(f"  {k}: {out[k].shape}  range {out[k].index.min().date()} -> {out[k].index.max().date()}")
    return out


def memory_dynamics(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Δd, rolling Vol(d), rolling Trend(d) — all evaluated on the rolling-stride
    cadence of d. Stride is roughly 5 trading days, so a 22-step horizon ≈ 110
    days; we use a smaller window of 4 (~20 trading days) for dynamics.
    """
    win = max(4, DYNAMICS_LAG // 5)
    delta = d.diff()
    vol = d.rolling(window=win, min_periods=2).std()

    def _slope(arr: np.ndarray) -> float:
        m = ~np.isnan(arr)
        if m.sum() < 2:
            return np.nan
        x = np.arange(len(arr))[m]
        return float(np.polyfit(x, arr[m], 1)[0])

    trend = d.rolling(window=win, min_periods=2).apply(_slope, raw=True)
    return {"delta": delta, "vol": vol, "trend": trend}


def cross_sectional_features(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "cs_mean_d": d.mean(axis=1),
        "cs_std_d": d.std(axis=1),
        "cs_median_d": d.median(axis=1),
        "cs_skew_d": d.apply(lambda r: stats.skew(r.dropna()), axis=1),
        "cs_kurt_d": d.apply(lambda r: stats.kurtosis(r.dropna()), axis=1),
        "cs_pct_above_30": (d > 0.30).mean(axis=1),
        "cs_range_d": d.max(axis=1) - d.min(axis=1),
    })


def sector_mean_panel(d: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """For each (date, ticker) return the mean d_t of all stocks in the same
    GICS sector — broadcast to the same shape as d."""
    sec_series = pd.Series(sectors)
    sec_series = sec_series.reindex(d.columns)
    out = pd.DataFrame(index=d.index, columns=d.columns, dtype=float)
    for sec, tickers in sec_series.groupby(sec_series).groups.items():
        if len(tickers) == 0:
            continue
        sec_mean = d[list(tickers)].mean(axis=1)
        for t in tickers:
            out[t] = sec_mean
    return out


def threshold_flags(d: pd.DataFrame, taus: tuple[float, ...] = THRESHOLDS) -> dict[float, pd.DataFrame]:
    return {tau: (d > tau).astype(float) for tau in taus}


def interaction_panels(d: pd.DataFrame, market_axis: pd.DataFrame,
                       liquidity: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each interaction is (T_sample x N): elementwise product of d with a
    market scalar, broadcast across stocks, or with per-stock liquidity proxy.
    """
    vix = market_axis["VIX"].reindex(d.index).ffill()
    move = market_axis["MOVE"].reindex(d.index).ffill()

    illiq = liquidity.reindex(d.index).clip(lower=LIQUIDITY_FLOOR)
    illiq_inv = 1.0 / illiq

    return {
        "d_x_vix": d.multiply(vix, axis=0),
        "d_x_move": d.multiply(move, axis=0),
        "d_x_illiq": d * illiq_inv,
    }


def build_har(rv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "har_d": rv.shift(HAR_DAILY),
        "har_w": rv.rolling(HAR_WEEKLY).mean().shift(1),
        "har_m": rv.rolling(HAR_MONTHLY).mean().shift(1),
    }


def liquidity_proxy(volume: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Average daily dollar volume over a 22-day window, used both as a
    descriptive feature and to construct the d × illiq interaction."""
    dollar_vol = (volume * prices).rolling(22).mean()
    return dollar_vol


def export_table4(panels: dict[str, pd.DataFrame],
                  cs: pd.DataFrame, mkt_axis: pd.DataFrame,
                  fp: Path) -> None:
    """Compact per-feature summary table grouped by category."""

    def _stats_panel(df: pd.DataFrame) -> dict:
        v = df.values.flatten()
        v = v[~np.isnan(v)]
        if v.size == 0:
            return dict(mean=np.nan, std=np.nan, p1=np.nan, p99=np.nan, n=0)
        return dict(mean=v.mean(), std=v.std(),
                    p1=np.percentile(v, 1), p99=np.percentile(v, 99), n=int(v.size))

    rows = []
    rows.append(("LRD", "$\\hat d_{GPH,t}$ (rolling)", _stats_panel(panels["d_gph"])))
    rows.append(("LRD", "$\\hat d_{LW,t}$ (rolling)", _stats_panel(panels["d_lw"])))
    rows.append(("Roughness", "$H_t$ (rolling)", _stats_panel(panels["h"])))
    rows.append(("Memory dyn.", "$\\Delta\\hat d_t$", _stats_panel(panels["delta_d_gph"])))
    rows.append(("Memory dyn.", "$\\mathrm{Vol}(\\hat d)_t$", _stats_panel(panels["vol_d_gph"])))
    rows.append(("Memory dyn.", "$\\mathrm{Trend}(\\hat d)_t$", _stats_panel(panels["trend_d_gph"])))
    rows.append(("HAR", "$RV^d$", _stats_panel(panels["har_d"])))
    rows.append(("HAR", "$RV^w$", _stats_panel(panels["har_w"])))
    rows.append(("HAR", "$RV^m$", _stats_panel(panels["har_m"])))
    rows.append(("Sector aggregate", "$\\bar d_{s(i),t}$", _stats_panel(panels["sector_mean_d"])))

    cs_summary = cs.describe().T[["mean", "std"]]
    cs_summary["p1"] = cs.quantile(0.01)
    cs_summary["p99"] = cs.quantile(0.99)
    cs_summary["n"] = cs.notna().sum()
    cs_label_map = {
        "cs_mean_d": "$\\bar d_t$ (cross-sectional mean)",
        "cs_std_d": "$\\sigma_d^t$ (cross-sectional std)",
        "cs_skew_d": "Skew of $d_t$ across stocks",
        "cs_kurt_d": "Kurt of $d_t$ across stocks",
        "cs_pct_above_30": "$P(\\hat d > 0.30)$",
        "cs_range_d": "Range of $\\hat d$ across stocks",
    }
    for k in ["cs_mean_d", "cs_std_d", "cs_skew_d", "cs_kurt_d", "cs_pct_above_30", "cs_range_d"]:
        if k in cs_summary.index:
            r = cs_summary.loc[k]
            rows.append(("Cross-sectional",
                         cs_label_map[k],
                         dict(mean=r["mean"], std=r["std"], p1=r["p1"], p99=r["p99"], n=int(r["n"]))))

    mkt_summary = mkt_axis.describe().T
    for k in ["VIX", "MOVE", "USYC2Y10"]:
        if k in mkt_summary.index:
            r = mkt_summary.loc[k]
            rows.append(("Market",
                         k,
                         dict(mean=r["mean"], std=r["std"],
                              p1=mkt_axis[k].quantile(0.01),
                              p99=mkt_axis[k].quantile(0.99),
                              n=int(mkt_axis[k].notna().sum()))))

    rows.append(("Interaction", "$\\hat d_t \\cdot \\mathrm{VIX}_t$",
                 _stats_panel(panels["d_x_vix"])))
    rows.append(("Interaction", "$\\hat d_t \\cdot \\mathrm{MOVE}_t$",
                 _stats_panel(panels["d_x_move"])))
    rows.append(("Interaction", "$\\hat d_t / \\mathrm{Liq}_t$",
                 _stats_panel(panels["d_x_illiq"])))

    with open(fp, "w") as f:
        f.write("% Table 4: Persistence Feature Vector — definitions and pooled statistics\n\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Persistence Feature Vector: Definitions and Pooled Statistics}\n")
        f.write("\\label{tab:features}\n\\small\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Category & Feature & Mean & Std & 1\\% & 99\\% \\\\\n\\midrule\n")
        prev_cat = None
        for cat, feat, s in rows:
            if cat != prev_cat:
                if prev_cat is not None:
                    f.write("\\midrule\n")
                f.write(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{cat}}}}} \\\\\n")
                prev_cat = cat
            f.write(f"  & {feat} & {s['mean']:.4f} & {s['std']:.4f} & "
                    f"{s['p1']:.4f} & {s['p99']:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\begin{tablenotes}\\small\n")
        f.write("\\item Notes: Statistics are pooled across all stocks and dates "
                "in the rolling-estimation panel. $\\hat d_t$ values are taken on a "
                "weekly stride; HAR components are constructed from Parkinson realised "
                "variance. Sector aggregates use GICS level-1; threshold indicators and "
                "interaction terms (with VIX, MOVE, and inverse dollar-volume liquidity) "
                "use the GPH estimate of $\\hat d_t$.\n")
        f.write("\\end{tablenotes}\n\\end{table}\n")


def save_panels(panels: dict[str, pd.DataFrame], cs: pd.DataFrame,
                mkt_axis: pd.DataFrame, thresholds: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in panels.items():
        df.to_csv(dest / f"feat_{name}.csv")
    cs.to_csv(dest / "feat_cross_section.csv")
    mkt_axis.to_csv(dest / "market_axes.csv")
    for tau, flag in thresholds.items():
        flag.to_csv(dest / f"feat_threshold_{int(tau*100):02d}.csv")
    print(f"\nSaved feature panels to {dest}")
    for fp in sorted(dest.glob("*.csv")):
        print(f"  {fp.name}")


def main() -> None:
    print("=" * 70)
    print("   MODULE 3: PERSISTENCE FEATURE ENGINEERING")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_clean_panel()
    sectors = sector_map(panel.metadata)
    print(f"  Panel: N={len(panel.kept)} T={len(panel.prices)}")

    print("\n[1] Loading Phase 2 outputs...")
    p2 = load_phase2()
    d_gph, d_lw, h = p2["d_gph"], p2["d_lw"], p2["h"]
    sample_dates = d_gph.index

    print("\n[2] Memory dynamics on d_GPH and d_LW...")
    dyn_gph = memory_dynamics(d_gph)
    dyn_lw = memory_dynamics(d_lw)
    dyn_h = memory_dynamics(h)

    print("\n[3] HAR components on Parkinson RV (forward-aligned to forecast next-day)...")
    har = build_har(panel.rv_parkinson)
    har_on_stride = {k: v.reindex(sample_dates).ffill() for k, v in har.items()}

    print("\n[4] Cross-sectional aggregates of d_GPH...")
    cs = cross_sectional_features(d_gph)

    print("\n[5] Sector-mean panel sector_mean_d_{s(i),t}...")
    sector_mean_d = sector_mean_panel(d_gph, sectors)

    print("\n[6] Threshold indicators 1{d_t > tau}...")
    thresholds = threshold_flags(d_gph, THRESHOLDS)

    print("\n[7] Market axis aligned to rolling stride...")
    mkt_cols = ["VIX", "MOVE", "USYC2Y10", "USGG10YR", "CDX_IG_5Y", "CDX_HY_5Y"]
    mkt_cols = [c for c in mkt_cols if c in panel.market.columns]
    mkt_axis = panel.market[mkt_cols].reindex(sample_dates).ffill()

    print("\n[8] Liquidity proxy (rolling 22d dollar volume) and interactions...")
    volume = pd.read_csv(BASE / "bloomberg_pull/processed/volume.csv",
                         index_col=0, parse_dates=True)
    volume = volume.reindex(panel.prices.index)[panel.kept]
    illiq_proxy = liquidity_proxy(volume, panel.prices).reindex(sample_dates).ffill()
    interactions = interaction_panels(d_gph, mkt_axis, illiq_proxy)

    panels = {
        "d_gph": d_gph, "d_lw": d_lw, "h": h,
        "delta_d_gph": dyn_gph["delta"], "vol_d_gph": dyn_gph["vol"], "trend_d_gph": dyn_gph["trend"],
        "delta_d_lw": dyn_lw["delta"],   "vol_d_lw": dyn_lw["vol"],   "trend_d_lw": dyn_lw["trend"],
        "delta_h": dyn_h["delta"],
        "har_d": har_on_stride["har_d"], "har_w": har_on_stride["har_w"], "har_m": har_on_stride["har_m"],
        "sector_mean_d": sector_mean_d,
        "d_x_vix": interactions["d_x_vix"],
        "d_x_move": interactions["d_x_move"],
        "d_x_illiq": interactions["d_x_illiq"],
    }

    print("\n[9] Saving panels and Table 4...")
    save_panels(panels, cs, mkt_axis, thresholds, FEAT_DIR)
    export_table4(panels, cs, mkt_axis, TABLES_DIR / "table4_features.tex")

    print("\n" + "=" * 70)
    print("Outputs:")
    print(f"  {TABLES_DIR/'table4_features.tex'}")
    print(f"  {FEAT_DIR}/  ({len(panels)+1+len(thresholds)+1} CSV panels)")

    print("\nFeature counts (per (T_sample x N) panel):")
    for k, df in panels.items():
        print(f"  {k:18s}  {df.shape}  non-null={df.notna().mean().mean()*100:.1f}%")
    print(f"  cs (T_sample x F)   {cs.shape}")
    print(f"  market axis         {mkt_axis.shape}")
    print(f"  threshold flags     {len(thresholds)} panels of shape {next(iter(thresholds.values())).shape}")


if __name__ == "__main__":
    main()
