"""
MODULE 22: Strict-Cutoff Timing Variant (run-26) and Roughness Discretization (run-27)
=====================================================================================
run-26  HAR-X with VIX and MOVE lagged one trading day (a strict 4:00 p.m.
        cutoff, since both indices are finalised after the equity close), vs
        the aligned HAR of module 4; and vs aligned HAR-X. Gives the value of
        implied volatility under the most conservative information set and
        the denominator for the "about a third" statement.
run-27  The range-noise simulation of module 16 at 390, 1560 and 6240 steps
        per day, with the fitted latent model held fixed; the theoretical
        variance of log(R^2) for the continuous range is reported alongside.

Outputs: results/intermediate/timing_strict.csv, roughness_discretization.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from modules.forecast_io import HORIZONS, build_targets, load_bundle
from modules.module6_forecast_eval import diebold_mariano, squared_loss
from modules.module15_timing import _load, stale_forecasts
from modules import module16_roughness as m16

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"


def timing_strict() -> pd.DataFrame:
    bundle = load_bundle()
    targets = build_targets(bundle)
    daily = bundle.panel.market[["VIX", "MOVE"]].ffill().shift(1).reindex(bundle.sample_dates)
    kept = bundle.panel.kept
    lagged = {"vix": pd.DataFrame({t: daily["VIX"] for t in kept}),
              "move": pd.DataFrame({t: daily["MOVE"] for t in kept})}
    rows = []
    for h in HORIZONS:
        yh_a, y_a = _load("A", h)
        yh_b, y_b = _load("A1", h)
        yh_l, y_l = stale_forecasts(bundle, "A1", h, lagged, targets)
        L = {"har": squared_loss(yh_a, y_a), "harx": squared_loss(yh_b, y_b), "harx_lag": squared_loss(yh_l, y_l)}
        mask = L["har"].notna() & L["harx"].notna() & L["harx_lag"].reindex_like(L["har"]).notna()
        L = {k: v.reindex_like(mask).where(mask) for k, v in L.items()}
        mse = {k: float(np.nanmean(v.values)) for k, v in L.items()}
        _, t1, p1, T = diebold_mariano(L["har"], L["harx_lag"], h=h)
        _, t2, p2, _ = diebold_mariano(L["harx_lag"], L["harx"], h=h)
        rows.append({"h": h, "T": T, "mse_har": mse["har"], "mse_harx_lagged_indices": mse["harx_lag"],
                     "mse_harx_aligned": mse["harx"],
                     "gain_lagged_vs_har_pct": 100 * (1 - mse["harx_lag"] / mse["har"]), "dm_t_lagged_vs_har": t1,
                     "gain_aligned_vs_har_pct": 100 * (1 - mse["harx"] / mse["har"]),
                     "gain_aligned_vs_lagged_pct": 100 * (1 - mse["harx"] / mse["harx_lag"]), "dm_t_aligned_vs_lagged": t2})
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "timing_strict.csv", index=False)
    return df


def roughness_discretization() -> pd.DataFrame:
    prev = pd.read_csv(INTERM / "roughness_null.csv").iloc[0]
    rng = np.random.default_rng(m16.SEED)
    T = 6136
    va = prev["total_var"] - prev["noise_var"]
    va_a, va_b = va * prev["share_a"], va * (1 - prev["share_a"])
    rows = []
    for steps in (390, 1560, 6240):
        m16.N_STEPS = steps
        eta = m16.range_noise(100_000, rng)
        hs = []
        for _ in range(100):
            lat = m16.simulate_ar(prev["phi_a"], va_a, T, rng) + m16.simulate_ar(prev["phi_b"], va_b, T, rng)
            noise = m16.range_noise(T, rng)
            hs.append(m16.hurst_scaling(lat + noise - noise.mean()))
        rows.append({"steps_per_day": steps, "noise_var": float(eta.var()), "noise_mean": float(eta.mean()),
                     "H_sim_observed_mean": float(np.mean(hs)), "H_sim_observed_sd": float(np.std(hs, ddof=1))})
    # continuous-range reference: Var[log R^2] = 4 Var[log R]; log R of a Brownian
    # range has variance pi^2/8 - ... (no closed form in common use); report the
    # finest discretization as the reference value instead.
    df = pd.DataFrame(rows)
    df.to_csv(INTERM / "roughness_discretization.csv", index=False)
    return df


def main() -> None:
    pd.set_option("display.width", 220)
    print("=" * 70); print("   MODULE 22: TIMING (STRICT CUTOFF) AND ROUGHNESS DISCRETIZATION"); print("=" * 70)
    print("\n[run-26]"); print(timing_strict().round(4).to_string(index=False))
    print("\n[run-27]"); print(roughness_discretization().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
