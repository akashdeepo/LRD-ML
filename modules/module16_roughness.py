"""
MODULE 16: Is Daily Range-Based Volatility Rough? A Non-Rough Null
=================================================================
The increment-scaling Hurst exponent of daily log Parkinson variance is about
0.06 (module 2), which read literally indicates very rough volatility. This
module asks whether a smooth (non-rough) process observed through the
Parkinson estimator gives the same number (ledger run-17, finding #15).

Latent log variance is the sum of two independent AR(1) components. The
observed series adds eta = log(R^2 / (4 ln 2)) - E[.], where R is the range
of a standard Brownian motion over one day sampled at 390 one-minute steps:
the sampling error of the Parkinson estimator, with no free parameter. The
AR parameters are fitted to the median autocorrelations and variance of the
observed series. Hurst exponents are estimated on simulated observed and
latent paths with module 2's estimator.

Outputs:
    results/intermediate/roughness_null.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.forecast_io import load_bundle
from modules.module2_lrd_estimation import hurst_scaling

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
INTERM = BASE / "results" / "intermediate"

LAGS = np.array([1, 2, 3, 5, 10, 22, 66, 125, 250])
N_STEPS = 390                 # one-minute steps in a trading day
N_REPS = 200
SEED = 0


def range_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draws of log(R^2 / (4 ln 2)) for the range R of a discretely sampled
    standard Brownian motion over one day (unit daily variance)."""
    out = np.empty(n)
    for k in range(0, n, 2000):
        m = min(2000, n - k)
        paths = np.cumsum(rng.standard_normal((m, N_STEPS)) / np.sqrt(N_STEPS), axis=1)
        paths = np.concatenate([np.zeros((m, 1)), paths], axis=1)
        r = paths.max(axis=1) - paths.min(axis=1)
        out[k:k + m] = np.log(r ** 2 / (4 * np.log(2)))
    return out


def acf(x: np.ndarray, lags: np.ndarray) -> np.ndarray:
    x = x[~np.isnan(x)]
    x = x - x.mean()
    v = (x ** 2).mean()
    return np.array([(x[k:] * x[:-k]).mean() / v for k in lags])


def fit_two_factor(target_acf: np.ndarray, total_var: float, noise_var: float) -> dict:
    """Fit (phi_a, phi_b, share_a) so that the ACF of latent + noise matches
    the target; latent variance = total_var - noise_var."""
    lat_var = total_var - noise_var

    def model_acf(p):
        pa, pb, s = 1 / (1 + np.exp(-p[0])), 1 / (1 + np.exp(-p[1])), 1 / (1 + np.exp(-p[2]))
        return (lat_var * (s * pa ** LAGS + (1 - s) * pb ** LAGS)) / total_var, pa, pb, s

    def loss(p):
        return float(((model_acf(p)[0] - target_acf) ** 2).sum())

    res = min((minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 4000})
               for x0 in ([1.0, 5.0, 0.0], [2.0, 6.0, 1.0], [0.0, 4.0, -1.0])),
              key=lambda r: r.fun)
    fitted, pa, pb, s = model_acf(res.x)
    return {"phi_a": pa, "phi_b": pb, "share_a": s, "latent_var": lat_var,
            "fit_sse": res.fun, "fitted_acf": fitted}


def simulate_ar(phi: float, var: float, T: int, rng: np.random.Generator) -> np.ndarray:
    sd = np.sqrt(var * (1 - phi ** 2))
    e = rng.standard_normal(T) * sd
    x = np.empty(T)
    x[0] = rng.standard_normal() * np.sqrt(var)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def main() -> None:
    print("=" * 70)
    print("   MODULE 16: ROUGHNESS NULL")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    log_rv = load_bundle().log_rv
    T = len(log_rv)
    acfs = np.array([acf(log_rv[c].values, LAGS) for c in log_rv.columns])
    target = np.median(acfs, axis=0)
    total_var = float(np.median(log_rv.var().values))
    h_data = np.array([hurst_scaling(log_rv[c].dropna().values) for c in log_rv.columns])

    eta = range_noise(200_000, rng)
    noise_var = float(eta.var())
    fit = fit_two_factor(target, total_var, noise_var)
    print(f"  data: var {total_var:.3f}, H mean {np.nanmean(h_data):.3f}; noise var {noise_var:.3f}")
    print(f"  fit: phi_a {fit['phi_a']:.4f} phi_b {fit['phi_b']:.4f} share_a {fit['share_a']:.3f} "
          f"SSE {fit['fit_sse']:.5f}")
    print("  acf data :", np.round(target, 3))
    print("  acf model:", np.round(fit["fitted_acf"], 3))

    h_obs, h_lat = [], []
    va, vb = fit["latent_var"] * fit["share_a"], fit["latent_var"] * (1 - fit["share_a"])
    for _ in range(N_REPS):
        lat = simulate_ar(fit["phi_a"], va, T, rng) + simulate_ar(fit["phi_b"], vb, T, rng)
        noise = range_noise(T, rng)
        obs = lat + noise - noise.mean()
        h_obs.append(hurst_scaling(obs))
        h_lat.append(hurst_scaling(lat))
    h_obs, h_lat = np.array(h_obs), np.array(h_lat)
    row = {"H_data_mean": np.nanmean(h_data), "H_data_p10": np.nanpercentile(h_data, 10),
           "H_data_p90": np.nanpercentile(h_data, 90),
           "H_sim_observed_mean": h_obs.mean(), "H_sim_observed_sd": h_obs.std(ddof=1),
           "H_sim_latent_mean": h_lat.mean(), "H_sim_latent_sd": h_lat.std(ddof=1),
           "phi_a": fit["phi_a"], "phi_b": fit["phi_b"], "share_a": fit["share_a"],
           "noise_var": noise_var, "total_var": total_var, "acf_fit_sse": fit["fit_sse"]}
    pd.DataFrame([row]).to_csv(INTERM / "roughness_null.csv", index=False)
    print(pd.Series(row).round(4).to_string())


if __name__ == "__main__":
    main()
