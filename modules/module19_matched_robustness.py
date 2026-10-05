"""
MODULE 19: Matched Robustness (run-22) and Balanced Panel / No-Hurst Variants (runs 28, 29)
==========================================================================================
Table 8 Panel A compared model C with HAR under each variant. The co-author
review asks for the comparison that matters: model C minus HAR-X on the same
cells under the same variant, with uncertainty. This module reuses the variant
builders of module 9 and adds:

  run-22  C vs HAR-X (h = 5) for: local Whittle d, windows 500 and 1000, the
          squared-return target (HAR-X refit on that target), and the two
          liquidity halves; bootstrap intervals (module 17).
  run-28  Balanced panel: HAR-X vs HAR and C vs HAR-X on the 107 stocks with
          complete history, h = 1, 5, 22 (saved forecasts, restricted cells).
  run-29  No-Hurst variant of C (16 predictors) vs HAR-X, h = 1, 5, 22.

Outputs: results/intermediate/matched_robustness.csv, balanced_panel.csv,
         no_hurst_variant.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import (
    HORIZONS, MODEL_FEATURES, aligned_xy, build_targets, load_bundle, stock_matrix,
)
from modules.module2_lrd_estimation import gph, hurst_scaling
from modules.module4_benchmarks import INIT_TRAIN_FRAC, expanding_forecast
from modules.module6_forecast_eval import diebold_mariano, squared_loss
from modules.module9_robustness import (
    H, _apply_d_overrides, _build_derived_from_d, _refit_model_C, _rolling_d_at_window,
)
from modules.module17_eval_checks import _block_boot_ratio

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
FCST = BASE / "results" / "intermediate" / "forecasts"
INTERM = BASE / "results" / "intermediate"
BB = BASE / "bloomberg_pull" / "processed"


def _load(model: str, h: int):
    return (pd.read_csv(FCST / f"{model}_h{h:02d}_yhat.csv", index_col=0, parse_dates=True),
            pd.read_csv(FCST / f"{model}_h{h:02d}_y.csv", index_col=0, parse_dates=True))


def compare(yh_s, y_s, yh_b, y_b, h, stocks=None, rng=None) -> dict:
    """Spec vs baseline on common cells: gain, DM, bootstrap interval."""
    if stocks is not None:
        cols = [c for c in stocks if c in yh_s.columns and c in yh_b.columns]
        yh_s, y_s, yh_b, y_b = yh_s[cols], y_s[cols], yh_b[cols], y_b[cols]
    Ls, Lb = squared_loss(yh_s, y_s), squared_loss(yh_b, y_b)
    Ls, Lb = Ls.align(Lb, join="inner")
    cm = Ls.notna() & Lb.notna()
    Ls, Lb = Ls.where(cm), Lb.where(cm)
    ms, mb = float(np.nanmean(Ls.values)), float(np.nanmean(Lb.values))
    _, t, p, T = diebold_mariano(Lb, Ls, h=h)
    out = {"mse_spec": ms, "mse_base": mb, "gain_pct": 100 * (1 - ms / mb), "dm_t": t, "dm_p": p, "T": T,
           "n_stocks": int(cm.any(axis=0).sum())}
    if rng is not None:
        lo, hi = _block_boot_ratio(Lb.mean(axis=1).dropna().values, Ls.mean(axis=1).dropna().values, rng)
        out["ci_lo"], out["ci_hi"] = lo, hi
    return out


def _fit(bundle, model: str, h: int, targets=None, features=None):
    """Per-stock expanding OLS for `model` (optionally with a custom predictor
    list) and optional custom targets; returns (yhat, y) panels."""
    sd = bundle.sample_dates
    init_date = sd[int(len(sd) * INIT_TRAIN_FRAC)]
    targets = targets or build_targets(bundle)
    if features is not None:
        MODEL_FEATURES[model] = features
    yh = pd.DataFrame(index=sd, columns=bundle.panel.kept, dtype=float); yy = yh.copy()
    for tkr in bundle.panel.kept:
        X, y = aligned_xy(stock_matrix(bundle, tkr, model, targets), h)
        if len(X) < 60:
            continue
        yh.loc[X.index, tkr] = expanding_forecast(X, y, init_date, h, bundle.rv.index).values
        yy.loc[y.index, tkr] = y.values
    return yh, yy


def sqret_targets(bundle) -> dict:
    sq = bundle.returns ** 2
    full = sq.index; sd = bundle.sample_dates
    pos = full.get_indexer(sd)
    out = {}
    for h in HORIZONS:
        arr = np.full((len(sd), sq.shape[1]), np.nan)
        for k, p in enumerate(pos):
            if p < 0 or p + h >= len(full):
                continue
            m = np.nanmean(sq.values[p + 1:p + 1 + h], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                arr[k] = np.where(m > 0, np.log(m), np.nan)
        out[h] = pd.DataFrame(arr, index=sd, columns=sq.columns)
    return out


def liquidity_halves(bundle):
    prc = pd.read_csv(BB / "prices_close.csv", index_col=0, parse_dates=True)
    vol = pd.read_csv(BB / "volume.csv", index_col=0, parse_dates=True)
    common = sorted(set(prc.columns) & set(vol.columns) & set(bundle.panel.kept))
    score = (1.0 / (prc[common] * vol[common]).rolling(22).mean()).mean(axis=0).dropna()
    med = score.median()
    return score[score >= med].index.tolist(), score[score < med].index.tolist()


# ------------------------------------------------------------------ run-22
def matched_robustness() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    yh_b, y_b = _load("A1", H)
    rows = []
    yh_c, y_c = _load("C", H)
    rows.append({"variant": "headline", **compare(yh_c, y_c, yh_b, y_b, H, rng=rng)})
    lo_liq, hi_liq = liquidity_halves(load_bundle())
    rows.append({"variant": "liquidity_high_illiq", **compare(yh_c, y_c, yh_b, y_b, H, stocks=lo_liq, rng=rng)})
    rows.append({"variant": "liquidity_low_illiq", **compare(yh_c, y_c, yh_b, y_b, H, stocks=hi_liq, rng=rng)})

    b = load_bundle()
    _apply_d_overrides(b, _build_derived_from_d(b.feat["d_lw"], b.sectors))
    yh, y = _refit_model_C(b)
    rows.append({"variant": "estimator_LW", **compare(yh, y, yh_b, y_b, H, rng=rng)})

    for W in (500, 1000):
        b = load_bundle()
        _apply_d_overrides(b, _build_derived_from_d(_rolling_d_at_window(b, W, gph), b.sectors))
        new_h = _rolling_d_at_window(b, W, hurst_scaling)
        b.feat["h"], b.feat["delta_h"] = new_h, new_h.diff()
        yh, y = _refit_model_C(b)
        rows.append({"variant": f"window_{W}", **compare(yh, y, yh_b, y_b, H, rng=rng)})

    b = load_bundle()
    tg = sqret_targets(b)
    yh_cs, y_cs = _fit(b, "C", H, targets=tg)
    yh_bs, y_bs = _fit(b, "A1", H, targets=tg)
    rows.append({"variant": "target_sqret", **compare(yh_cs, y_cs, yh_bs, y_bs, H, rng=rng)})

    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "matched_robustness.csv", index=False)
    return df


# ------------------------------------------------------------------ run-28
def balanced_panel() -> pd.DataFrame:
    prc = pd.read_csv(BB / "prices_close.csv", index_col=0, parse_dates=True)
    kept = open(BB / "clean_panel" / "kept_tickers.txt").read().split()
    cov = prc[kept].notna().mean()
    full_hist = cov[cov >= 0.995].index.tolist()
    rng = np.random.default_rng(1)
    rows = []
    for h in HORIZONS:
        yh_a, y_a = _load("A", h); yh_b, y_b = _load("A1", h); yh_c, y_c = _load("C", h)
        for name, (s, b) in {"HAR-X vs HAR": ((yh_b, y_b), (yh_a, y_a)),
                             "C vs HAR-X": ((yh_c, y_c), (yh_b, y_b))}.items():
            for panel, stocks in (("all 115", None), ("balanced 107", full_hist)):
                rows.append({"comparison": name, "panel": panel, "h": h,
                             **compare(s[0], s[1], b[0], b[1], h, stocks=stocks, rng=rng)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "balanced_panel.csv", index=False)
    return df


# ------------------------------------------------------------------ run-29
def no_hurst_variant() -> pd.DataFrame:
    b = load_bundle()
    feats = [f for f in MODEL_FEATURES["C"] if f not in ("h", "delta_h")]
    rng = np.random.default_rng(2)
    rows = []
    for h in HORIZONS:
        yh, y = _fit(b, "C_noH", h, features=feats)
        yh_b, y_b = _load("A1", h)
        yh_c, y_c = _load("C", h)
        rows.append({"spec": "C without Hurst features (16)", "h": h, **compare(yh, y, yh_b, y_b, h, rng=rng)})
        rows.append({"spec": "C (18)", "h": h, **compare(yh_c, y_c, yh_b, y_b, h, rng=rng)})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "no_hurst_variant.csv", index=False)
    return df


def main() -> None:
    pd.set_option("display.width", 220)
    print("=" * 70); print("   MODULE 19: MATCHED ROBUSTNESS, BALANCED PANEL, NO-HURST"); print("=" * 70)
    print("\n[run-22] C vs HAR-X under each variant (h=5)"); print(matched_robustness().round(3).to_string(index=False))
    print("\n[run-28] balanced panel"); print(balanced_panel().round(3).to_string(index=False))
    print("\n[run-29] no-Hurst variant"); print(no_hurst_variant().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
