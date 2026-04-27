"""
forecast_io.py — assemble per-stock feature matrices and forecast targets
for Module 4 (linear benchmarks) and Module 5 (ML models).

Design choices:
  * Forecasting cadence = the rolling-estimation stride from Phase 2 (weekly).
    Each row in the resulting matrix corresponds to one sample date in the
    panels under results/intermediate/features/.
  * Target at sample date t for horizon h:
        y_{t,h} = log( mean( RV^{PK}_{t+1 .. t+h} ) )
    where the inner mean is over actual trading days, not sample dates.
  * All right-hand-side features at sample date t use only information
    available at t (no leakage).
  * Models are nested:
        A: HAR-RV core (log_RV_d, log_RV_w, log_RV_m, lagged_return)
        B: A + own-stock persistence (d_GPH, delta_d, vol_d, trend_d, H, delta_H)
        C: B + cross-sectional (cs_mean_d, cs_std_d, sector_mean_d) + market (VIX, MOVE)
        D: same predictors as C, but estimated by non-linear ML (Module 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from modules.io_v2 import build_clean_panel, sector_map

BASE = Path(__file__).resolve().parent.parent
FEAT = BASE / "results" / "intermediate" / "features"

HORIZONS = (1, 5, 22)

MODEL_FEATURES: dict[str, list[str]] = {
    "A": [
        "har_d_log", "har_w_log", "har_m_log",
        "ret_lag1", "ret_lag1_abs",
    ],
    "B": [
        "har_d_log", "har_w_log", "har_m_log",
        "ret_lag1", "ret_lag1_abs",
        "d_gph", "delta_d_gph", "vol_d_gph", "trend_d_gph",
        "h", "delta_h",
    ],
    "C": [
        "har_d_log", "har_w_log", "har_m_log",
        "ret_lag1", "ret_lag1_abs",
        "d_gph", "delta_d_gph", "vol_d_gph", "trend_d_gph",
        "h", "delta_h",
        "cs_mean_d", "cs_std_d", "sector_mean_d",
        "vix", "move",
        "d_x_vix", "d_x_move",
    ],
}
MODEL_FEATURES["D"] = MODEL_FEATURES["C"]  # same predictors, non-linear estimator


@dataclass
class StockMatrix:
    ticker: str
    sample_dates: pd.DatetimeIndex
    X: pd.DataFrame              # (T_sample x F)
    y: dict[int, pd.Series]      # h -> realized log RV averaged over next h days
    sector: str
    size_bucket: str


@dataclass
class Bundle:
    panel: object                 # io_v2.Panel
    sectors: dict
    sample_dates: pd.DatetimeIndex
    rv: pd.DataFrame              # full daily Parkinson RV (T x N)
    log_rv: pd.DataFrame          # full daily log RV
    returns: pd.DataFrame         # full daily winsorized log returns
    feat: dict[str, pd.DataFrame] # name -> (T_sample x N)
    cs: pd.DataFrame              # (T_sample x F)
    market: pd.DataFrame          # (T_sample x F) — already on sample stride


def _load_feat() -> dict[str, pd.DataFrame]:
    files = sorted(FEAT.glob("feat_*.csv"))
    return {
        f.stem.replace("feat_", ""): pd.read_csv(f, index_col=0, parse_dates=True)
        for f in files
    }


def load_bundle() -> Bundle:
    panel = build_clean_panel()
    sectors = sector_map(panel.metadata)
    feat = _load_feat()
    cs = pd.read_csv(FEAT / "feat_cross_section.csv", index_col=0, parse_dates=True)
    mkt = pd.read_csv(FEAT / "market_axes.csv", index_col=0, parse_dates=True)
    sample_dates = feat["d_gph"].index
    return Bundle(
        panel=panel,
        sectors=sectors,
        sample_dates=sample_dates,
        rv=panel.rv_parkinson,
        log_rv=panel.log_rv,
        returns=panel.returns,
        feat=feat,
        cs=cs,
        market=mkt,
    )


def _rv_target(rv_full: pd.DataFrame, sample_dates: pd.DatetimeIndex,
               h: int) -> pd.DataFrame:
    """For each (sample_date t, ticker), compute log( mean(RV_{t+1..t+h}) )
    using actual daily Parkinson RV (variance-scale)."""
    full_dates = rv_full.index
    rv_arr = rv_full.values
    out = np.full((len(sample_dates), rv_full.shape[1]), np.nan)
    pos = full_dates.get_indexer(sample_dates)
    for k, p in enumerate(pos):
        if p < 0 or p + h >= len(full_dates):
            continue
        block = rv_arr[p + 1: p + 1 + h, :]  # next h trading days
        mean = np.nanmean(block, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k, :] = np.where(mean > 0, np.log(mean), np.nan)
    return pd.DataFrame(out, index=sample_dates, columns=rv_full.columns)


def build_targets(bundle: Bundle) -> dict[int, pd.DataFrame]:
    return {h: _rv_target(bundle.rv, bundle.sample_dates, h) for h in HORIZONS}


def stock_matrix(bundle: Bundle, ticker: str, model: str,
                 targets: dict[int, pd.DataFrame]) -> StockMatrix:
    """Assemble the (T_sample x F) feature matrix and dict of horizon-h targets
    for one stock under one model spec."""
    feat = bundle.feat
    cs = bundle.cs
    mkt = bundle.market
    rv = bundle.rv

    har_d = feat["har_d"][ticker]
    har_w = feat["har_w"][ticker]
    har_m = feat["har_m"][ticker]
    har_d_log = np.log(har_d.where(har_d > 0))
    har_w_log = np.log(har_w.where(har_w > 0))
    har_m_log = np.log(har_m.where(har_m > 0))

    # lagged-return feature evaluated at the trading day before each sample date
    ret = bundle.returns[ticker]
    sample_dates = bundle.sample_dates
    pos = ret.index.get_indexer(sample_dates)
    lag_idx = np.where(pos > 0, pos - 1, -1)
    lag_vals = np.where(lag_idx >= 0, ret.values[lag_idx], np.nan)
    ret_lag1 = pd.Series(lag_vals, index=sample_dates, dtype=float)
    ret_lag1_abs = ret_lag1.abs()

    sources = {
        "har_d_log": har_d_log, "har_w_log": har_w_log, "har_m_log": har_m_log,
        "ret_lag1": ret_lag1, "ret_lag1_abs": ret_lag1_abs,
        "d_gph": feat["d_gph"][ticker], "delta_d_gph": feat["delta_d_gph"][ticker],
        "vol_d_gph": feat["vol_d_gph"][ticker], "trend_d_gph": feat["trend_d_gph"][ticker],
        "h": feat["h"][ticker], "delta_h": feat["delta_h"][ticker],
        "cs_mean_d": cs["cs_mean_d"], "cs_std_d": cs["cs_std_d"],
        "sector_mean_d": feat["sector_mean_d"][ticker],
        "vix": mkt["VIX"], "move": mkt["MOVE"],
        "d_x_vix": feat["d_x_vix"][ticker], "d_x_move": feat["d_x_move"][ticker],
    }

    cols = MODEL_FEATURES[model]
    X = pd.DataFrame({c: sources[c] for c in cols}, index=sample_dates)
    y = {h: targets[h][ticker] for h in HORIZONS}

    sector = bundle.sectors.get(ticker, "Other")
    meta = bundle.panel.metadata
    size = meta.set_index("BloombergTicker").loc[ticker, "SizeBucket"] if ticker in meta["BloombergTicker"].values else "Unknown"
    return StockMatrix(ticker=ticker, sample_dates=sample_dates,
                       X=X, y=y, sector=sector, size_bucket=size)


def aligned_xy(sm: StockMatrix, h: int) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with any NaN in features or target."""
    df = sm.X.copy()
    df["__y__"] = sm.y[h]
    df = df.dropna()
    return df.drop(columns="__y__"), df["__y__"]


if __name__ == "__main__":
    b = load_bundle()
    print(f"Bundle: N={len(b.panel.kept)}  T_sample={len(b.sample_dates)}")
    print(f"  sample range {b.sample_dates.min().date()} -> {b.sample_dates.max().date()}")
    targets = build_targets(b)
    for h, t in targets.items():
        cov = t.notna().mean().mean()
        print(f"  target h={h:2d}: shape {t.shape}, non-null {cov*100:.1f}%")
    for m in ("A", "B", "C"):
        sm = stock_matrix(b, "AAPL", m, targets)
        for h in (1, 5, 22):
            X, y = aligned_xy(sm, h)
            print(f"  model {m}, AAPL, h={h}: X={X.shape}, y={len(y)}")
