"""
MODULE 20: Window-Inclusion Additions and Memory-Estimation Checks (runs 23, 24)
================================================================================
Additions requested in the co-author review:

  run-24  (a) A null WITH long memory: ARFIMA(0,d,0) with d = median calm-period
          (2012-2019) local Whittle estimate across stocks and the calm-period
          innovation variance, plus the observed 2020 episode; 200 paths at
          W = 500, 750, 1000; mean path and 5-95% bands for this null and for
          the short-memory ARMA(1,1) null of module 14.
          (b) R^2 of the state on alternative window summaries of the VIX:
          max, mean, 90th percentile, and episode length (days with VIX > 30).
  run-23  (a) Bandwidth sensitivity: rolling GPH cross-sectional mean at W=750
          with m = T^0.5, T^0.6, T^0.65, T^0.7; correlation with the 0.65 series
          and the 2020 entry/exit changes.
          (b) Estimator validation: GPH and local Whittle on ARFIMA(0,d,0),
          d in {0, 0.2, 0.4}, T = 750, 200 replications, bandwidth T^0.65.
          (c) Missing days: rolling GPH with missing log-variance days linearly
          interpolated (inside each stock's history) vs dropped.

Outputs: results/intermediate/window_null_bands.csv, window_r2_alternatives.csv,
         bandwidth_sensitivity.csv, estimator_validation.csv, missing_days_sensitivity.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import load_bundle
from modules.module2_lrd_estimation import ROLLING_STRIDE, gph, local_whittle, rolling_panel
from modules.module14_window_inclusion import (
    CALM, N_REPS, WINDOWS, calm_arma, episode_profile, pick_events, step_change,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"
SEED = 0
FRAC_LAGS = 3000


def frac_diff_weights(d: float, n: int) -> np.ndarray:
    """Weights of (1-L)^{-d} up to lag n (MA(infinity) representation)."""
    w = np.empty(n + 1); w[0] = 1.0
    for k in range(1, n + 1):
        w[k] = w[k - 1] * (k - 1 + d) / k
    return w


def simulate_arfima(d: float, sigma: float, T: int, rng, burn: int = 1000) -> np.ndarray:
    w = frac_diff_weights(d, FRAC_LAGS)
    e = rng.standard_normal(T + burn + FRAC_LAGS) * sigma
    x = np.convolve(e, w, mode="full")[FRAC_LAGS:FRAC_LAGS + T + burn]
    return x[burn:]


def simulate_arma(ar: float, ma: float, s: float, T: int, rng, burn: int = 500) -> np.ndarray:
    e = rng.standard_normal(T + burn) * s
    x = np.empty(T + burn); x[0] = e[0]
    for t in range(1, T + burn):
        x[t] = ar * x[t - 1] + e[t] + ma * e[t - 1]
    return x[burn:]


def rolling_gph_path(x: np.ndarray, W: int, pre: int) -> tuple[np.ndarray, np.ndarray]:
    ends = np.arange(W - 1, len(x), ROLLING_STRIDE)
    return ends - pre, np.array([gph(x[e - W + 1:e + 1])[0] for e in ends])


def null_bands(log_rv: pd.DataFrame, profile: np.ndarray, arma, d_lm: float, sigma_lm: float) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for W in WINDOWS:
        pre = W + 120; T = pre + W + 300
        for kind in ("short_memory_arma", "long_memory_arfima"):
            paths = []
            for _ in range(N_REPS):
                x = (simulate_arma(*arma, T, rng) if kind == "short_memory_arma"
                     else simulate_arfima(d_lm, sigma_lm, T, rng))
                x = x.copy(); x[pre:pre + len(profile)] += profile
                rel, p = rolling_gph_path(x, W, pre)
                paths.append(p)
            P = np.array(paths)
            mean, lo, hi = P.mean(0), np.percentile(P, 5, 0), np.percentile(P, 95, 0)
            steps = np.array([step_change(mean, rel, int(e)) for e in rel])
            base = mean[(rel >= -120) & (rel < 0)].mean(); plat = mean[(rel >= 60) & (rel < W - 60)].mean()
            after = mean[(rel >= W + 20) & (rel < W + 120)].mean()
            for r, m, l, h in zip(rel, mean, lo, hi):
                rows.append({"W": W, "null": kind, "rel_day": int(r), "mean": m, "p05": l, "p95": h,
                             "baseline": base, "plateau": plat, "after_exit": after,
                             "rise": plat - base, "largest_drop_rel_day": int(rel[np.nanargmin(steps)])})
            print(f"  W={W} {kind}: baseline {base:.3f} plateau {plat:.3f} after {after:.3f} "
                  f"rise {plat - base:+.3f} largest drop day {int(rel[np.nanargmin(steps)])}")
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "window_null_bands.csv", index=False)
    return df


def r2_alternatives(log_rv: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    rows = []
    for W in WINDOWS:
        dbar = rolling_panel(log_rv, gph, window=W, stride=ROLLING_STRIDE, label=f"GPH W={W}").mean(axis=1)
        alt = {
            "max_vix": vix.rolling(W, min_periods=W // 2).max(),
            "mean_vix": vix.rolling(W, min_periods=W // 2).mean(),
            "p90_vix": vix.rolling(W, min_periods=W // 2).quantile(0.9),
            "days_vix_above_30": (vix > 30).astype(float).rolling(W, min_periods=W // 2).sum(),
            "current_vix": vix,
        }
        for name, s in alt.items():
            df = pd.DataFrame({"d": dbar, "x": s.reindex(dbar.index)}).dropna()
            rows.append({"W": W, "summary": name, "r2": np.corrcoef(df["d"], df["x"])[0, 1] ** 2})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "window_r2_alternatives.csv", index=False)
    return df


def bandwidth_sensitivity(log_rv: pd.DataFrame, covid: pd.Timestamp) -> pd.DataFrame:
    idx = log_rv.index; W = 750; p = idx.get_loc(covid)
    series = {}
    for power in (0.5, 0.6, 0.65, 0.7):
        est = (lambda pw: (lambda x: gph(x, m=int(np.floor(len(x) ** pw)))))(power)
        series[power] = rolling_panel(log_rv, est, window=W, stride=ROLLING_STRIDE, label=f"GPH m=T^{power}").mean(axis=1)
    ref = series[0.65]
    rows = []
    for power, s in series.items():
        ends = idx.get_indexer(s.index); v = s.values
        rows.append({"bandwidth_power": power, "m_at_750": int(np.floor(750 ** power)),
                     "mean_level": float(np.nanmean(v)), "corr_with_0.65": float(s.corr(ref)),
                     "entry_change_2020": step_change(v, ends, p), "exit_change_2020": step_change(v, ends, p + W)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "bandwidth_sensitivity.csv", index=False)
    return df


def estimator_validation() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for d in (0.0, 0.2, 0.4):
        g, l = [], []
        for _ in range(200):
            x = simulate_arfima(d, 1.0, 750, rng)
            g.append(gph(x)[0]); l.append(local_whittle(x)[0])
        rows.append({"true_d": d, "gph_mean": np.mean(g), "gph_sd": np.std(g, ddof=1),
                     "lw_mean": np.mean(l), "lw_sd": np.std(l, ddof=1)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "estimator_validation.csv", index=False)
    return df


def missing_days_sensitivity(log_rv: pd.DataFrame, covid: pd.Timestamp) -> pd.DataFrame:
    idx = log_rv.index; W = 750; p = idx.get_loc(covid)
    filled = log_rv.interpolate(limit_area="inside")
    s_drop = rolling_panel(log_rv, gph, window=W, stride=ROLLING_STRIDE, label="GPH drop-missing").mean(axis=1)
    s_fill = rolling_panel(filled, gph, window=W, stride=ROLLING_STRIDE, label="GPH interpolated").mean(axis=1)
    miss = log_rv.isna().mean()
    rows = []
    for name, s in (("drop_missing", s_drop), ("interpolate_missing", s_fill)):
        ends = idx.get_indexer(s.index); v = s.values
        rows.append({"treatment": name, "mean_level": float(np.nanmean(v)),
                     "corr_between": float(s_drop.corr(s_fill)),
                     "entry_change_2020": step_change(v, ends, p), "exit_change_2020": step_change(v, ends, p + W),
                     "missing_share_mean": float(miss.mean()), "missing_share_max": float(miss.max()),
                     "stocks_missing_over_5pct": int((miss > 0.05).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "missing_days_sensitivity.csv", index=False)
    return df


def main() -> None:
    pd.set_option("display.width", 200)
    print("=" * 70); print("   MODULE 20: WINDOW CHECKS"); print("=" * 70)
    bundle = load_bundle()
    log_rv = bundle.log_rv
    vix = bundle.panel.market["VIX"].reindex(log_rv.index).ffill()
    events = pick_events(log_rv)
    covid = [d for d in events if d.year == 2020][0]

    # calm-period long-memory parameters
    d_calm, s_calm = [], []
    for c in log_rv.columns:
        x = log_rv[c].loc[CALM[0]:CALM[1]].dropna().values
        if len(x) >= 1000:
            d_calm.append(local_whittle(x)[0]); s_calm.append(x.std())
    d_lm = float(np.median(d_calm)); sd_lm = float(np.median(s_calm))
    # innovation sd so the ARFIMA has the calm-period variance (approx. via simulation)
    rng = np.random.default_rng(SEED)
    sim_sd = simulate_arfima(d_lm, 1.0, 20000, rng).std()
    sigma_lm = sd_lm / sim_sd
    arma = calm_arma(log_rv)
    print(f"  calm d (local Whittle median) {d_lm:.3f}; calm sd {sd_lm:.3f}; ARFIMA innovation sd {sigma_lm:.3f}")
    profile = episode_profile(log_rv, covid)

    print("\n[run-24a] null bands"); null_bands(log_rv, profile, arma, d_lm, sigma_lm)
    print("\n[run-24b] R2 alternatives"); print(r2_alternatives(log_rv, vix).round(3).to_string(index=False))
    print("\n[run-23a] bandwidth"); print(bandwidth_sensitivity(log_rv, covid).round(3).to_string(index=False))
    print("\n[run-23b] estimator validation"); print(estimator_validation().round(3).to_string(index=False))
    print("\n[run-23c] missing days"); print(missing_days_sensitivity(log_rv, covid).round(3).to_string(index=False))
    pd.DataFrame([{"d_calm_lw_median": d_lm, "calm_sd": sd_lm, "arfima_innovation_sd": sigma_lm,
                   "arma_ar": arma[0], "arma_ma": arma[1], "arma_sigma": arma[2]}]).to_csv(
        INTERM / "window_null_params.csv", index=False)


if __name__ == "__main__":
    main()
