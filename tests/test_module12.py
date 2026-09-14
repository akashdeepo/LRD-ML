"""Sanity tests for the Clark-West, Holm and Ledoit-Wolf implementations in
module 12. Run with:  python -m pytest tests -q"""
import numpy as np
import pandas as pd
import pytest

from modules.module12_incremental_tests import (
    _hac_mean_test, _sharpe_diff_hac, clark_west, holm,
)


def _panels(T=400, N=30, seed=1, extra_signal=0.0):
    """Nested pair: small model = mean forecast; big model adds a regressor
    that carries `extra_signal` true signal plus estimation noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    cols = [f"s{i}" for i in range(N)]
    x = rng.standard_normal((T, N))
    y = extra_signal * x + rng.standard_normal((T, N))
    yhat_small = np.zeros((T, N))
    yhat_big = (extra_signal + 0.15 * rng.standard_normal((T, N))) * x   # noisy estimate
    return (pd.DataFrame(yhat_small, idx, cols), pd.DataFrame(yhat_big, idx, cols),
            pd.DataFrame(y, idx, cols))


def test_clark_west_null_is_centred():
    # no true signal: the big model only adds estimation noise; CW mean ~ 0
    ts = []
    for seed in range(20):
        s, b, y = _panels(seed=seed, extra_signal=0.0)
        ts.append(clark_west(s, b, y, h=1)["cw_t"])
    assert abs(np.mean(ts)) < 0.6          # not systematically positive/negative
    # DM on the raw loss differential would be biased against the big model
    d = ((y - s) ** 2 - (y - b) ** 2).mean(axis=1).values
    assert _hac_mean_test(d, 1)[2] < 0


def test_clark_west_detects_signal():
    s, b, y = _panels(seed=3, extra_signal=0.5)
    assert clark_west(s, b, y, h=1)["cw_t"] > 5


def test_holm_matches_statsmodels_when_available():
    p = [0.01, 0.04, 0.03, 0.20, 0.005]
    ours = holm(p)
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        pytest.skip("statsmodels not installed")
    ref = multipletests(p, method="holm")[1]
    assert np.allclose(ours, ref)


def test_holm_monotone_and_bounded():
    adj = holm([0.5, 0.9, 0.7])
    assert all(0 <= a <= 1 for a in adj)
    assert adj[np.argmax([0.5, 0.9, 0.7])] == max(adj)


def test_sharpe_diff_zero_for_identical_series():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(500) * 0.02 + 0.001
    d, se, t = _sharpe_diff_hac(r, r, bw=0)
    assert d == pytest.approx(0.0)


def test_sharpe_diff_sign_and_scale():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(3000) * 0.02
    r1 = base + 0.004      # higher mean, same vol -> higher Sharpe
    r2 = base + 0.001
    d, se, t = _sharpe_diff_hac(r1, r2, bw=0)
    assert d > 0 and t > 3
    # delta equals the difference of sample Sharpe ratios (population sd)
    sr = lambda r: r.mean() / r.std(ddof=0)
    assert d == pytest.approx(sr(r1) - sr(r2), rel=1e-9)
