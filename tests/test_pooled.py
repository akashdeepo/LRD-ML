"""The within-transformation fixed-effects fit in module 13 must equal OLS on
explicit group dummies, and absent groups must get a nan intercept."""
import numpy as np
import pytest

from modules.module13_candidates import _fe_fit


def test_fe_fit_equals_dummy_ols():
    rng = np.random.default_rng(0)
    n, p, K = 400, 3, 6
    codes = rng.integers(0, K - 1, size=n)          # group K-1 never appears
    X = rng.standard_normal((n, p))
    beta_true = np.array([0.5, -1.0, 2.0])
    alpha_true = np.linspace(-1, 1, K)
    y = alpha_true[codes] + X @ beta_true + 0.1 * rng.standard_normal(n)
    beta, alpha = _fe_fit(X, y, codes, K)
    D = np.zeros((n, K - 1)); D[np.arange(n), codes] = 1.0
    coef, *_ = np.linalg.lstsq(np.column_stack([X, D]), y, rcond=None)
    assert np.allclose(beta, coef[:p], atol=1e-8)
    assert np.allclose(alpha[: K - 1], coef[p:], atol=1e-8)
    assert np.isnan(alpha[K - 1])
    assert np.allclose(beta, beta_true, atol=0.05)
