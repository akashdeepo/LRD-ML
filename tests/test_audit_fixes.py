"""Regression tests for the 2026-10-02 audit fixes (issues I1, I4, I10, I11,
I12, I14, I20, I23, I6, I32). Run with:  python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest

from modules.forecast_io import first_eval_row, n_train_rows, pit_level_correction, row_positions
from modules.io_v2 import log_variance
from modules.module3_feature_engineering import build_har, sector_mean_panel
from modules.module4_benchmarks import expanding_forecast
from modules.module6_forecast_eval import diebold_mariano, hac_bandwidth, regime_masks


def test_har_uses_day_t():                                   # I1
    idx = pd.bdate_range("2020-01-01", periods=40)
    rv = pd.DataFrame({"X": np.arange(1.0, 41.0)}, index=idx)
    har = build_har(rv)
    t = idx[30]
    assert har["har_d"].loc[t, "X"] == rv.loc[t, "X"]
    assert har["har_w"].loc[t, "X"] == pytest.approx(rv["X"].iloc[26:31].mean())
    assert har["har_m"].loc[t, "X"] == pytest.approx(rv["X"].iloc[9:31].mean())


def test_log_variance_keeps_missing_and_zero_as_nan():       # I4
    rv = pd.DataFrame({"X": [0.0, np.nan, 1e-4, 4e-4]})
    lv = log_variance(rv)["X"]
    assert np.isnan(lv.iloc[0]) and np.isnan(lv.iloc[1])
    assert lv.iloc[2] == pytest.approx(np.log(1e-4))


def test_first_eval_row_anchors_on_date():                   # I23
    idx = pd.date_range("2010-01-01", periods=200, freq="7D")
    assert first_eval_row(idx, idx[120]) == 120
    assert first_eval_row(idx, idx[10]) == 52                  # min training rows


def test_expanding_forecast_start_and_no_lookahead():        # I23 + embargo
    rng = np.random.default_rng(0)
    full = pd.bdate_range("2010-01-01", periods=1500)
    idx = full[::5][:250]
    X = pd.DataFrame({"x": rng.standard_normal(len(idx))}, index=idx)
    y = pd.Series(0.5 * X["x"].values + rng.standard_normal(len(idx)), index=idx)
    init = idx[150]
    yh = expanding_forecast(X, y, init, 22, full)
    assert yh.iloc[:150].isna().all() and yh.iloc[150:].notna().all()
    # perturbing targets that end after origin t must not change the forecast at t
    t = 200
    n = n_train_rows(row_positions(idx, full), t, 22)
    y2 = y.copy(); y2.iloc[n:] += 100.0
    assert expanding_forecast(X, y2, init, 22, full).iloc[t] == pytest.approx(yh.iloc[t])


def test_pit_level_correction_uses_only_past_errors():       # I12, I14
    idx = pd.date_range("2010-01-01", periods=120, freq="7D")
    yhat = pd.DataFrame({"X": np.zeros(120)}, index=idx)
    y = pd.DataFrame({"X": np.full(120, 0.3)}, index=idx)
    out = pit_level_correction(yhat, y, h=22, kind="mean", min_obs=26)
    lag = 5
    assert out["X"].iloc[: 26 + lag - 1].isna().all()   # 26 usable errors first exist at row 30
    assert out["X"].iloc[60] == pytest.approx(0.3)
    y2 = y.copy(); y2.iloc[61:] = 9.0                          # future errors
    assert pit_level_correction(yhat, y2, 22, kind="mean")["X"].iloc[60] == pytest.approx(0.3)
    sm = pit_level_correction(yhat, y, 5, kind="smear")
    assert sm["X"].iloc[60] == pytest.approx(0.3)              # log mean exp of a constant


def test_sector_mean_is_leave_one_out():                     # I20
    idx = pd.date_range("2020-01-01", periods=3)
    d = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.3, 0.4, np.nan], "c": [0.5, 0.6, 0.7]}, index=idx)
    out = sector_mean_panel(d, {"a": "S", "b": "S", "c": "S"})
    assert out.loc[idx[0], "a"] == pytest.approx(0.4)          # mean of b, c
    assert out.loc[idx[2], "a"] == pytest.approx(0.7)          # b missing
    assert out.loc[idx[2], "b"] == pytest.approx(0.5)          # own missing: mean of a, c


def test_dm_matches_statsmodels_hac():                       # I11
    sm = pytest.importorskip("statsmodels.stats.sandwich_covariance")
    import statsmodels.api as sma
    rng = np.random.default_rng(3)
    T, N = 300, 20
    idx = pd.date_range("2010-01-01", periods=T, freq="7D")
    a = pd.DataFrame(rng.standard_normal((T, N)) ** 2 + 0.05, index=idx)
    b = pd.DataFrame(rng.standard_normal((T, N)) ** 2, index=idx)
    _, t_hln, _, T_ = diebold_mariano(a, b, h=5)
    d = (a - b).mean(axis=1).values
    bw = hac_bandwidth(T, 5)
    res = sma.OLS(d, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": bw, "use_correction": False})
    h_eff = 1
    hln = np.sqrt((T + 1 - 2 * h_eff + h_eff * (h_eff - 1) / T) / T)
    assert t_hln == pytest.approx(res.tvalues[0] * hln, rel=1e-6)
    assert bw == 5 and hac_bandwidth(645, 22) == 6 and hac_bandwidth(645, 1) == 6


def test_regime_quartiles_use_reference_dates():             # I10
    idx = pd.date_range("2000-01-01", periods=100, freq="7D")
    vix = pd.Series(np.r_[np.full(50, 50.0), np.linspace(10, 30, 50)], index=idx)
    m = regime_masks(idx, pd.DataFrame({"VIX": vix}), ref_idx=idx[50:])
    assert m["High VIX (Q4)"][:50].all()                       # all early dates above the ref Q3
    assert "GFC (2008-Q3 to 2009-Q4)" not in m


def test_tuned_gbm_chooses_from_grid():                      # I32
    from modules.module5_ml_models import GBM_GRID, GBM_TREES, TunedLGBM
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    y = 0.5 * X[:, 0] + rng.standard_normal(300)
    m = TunedLGBM(h=5).fit(X, y)
    assert {k: m.params_[k] for k in ("num_leaves", "min_data_in_leaf")} in GBM_GRID
    assert m.params_["n_estimators"] in GBM_TREES
    assert m.predict(X[:3]).shape == (3,)


def test_turnover_zero_for_static_unlevered_book_without_returns():   # I6
    from modules.module11_economic import turnover
    idx = pd.date_range("2020-01-01", periods=10, freq="7D")
    w = pd.DataFrame(1.0, index=idx, columns=["a", "b"])
    R = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    assert turnover(w, R, pd.Series(0.0, index=idx)).abs().max() == pytest.approx(0.0)
