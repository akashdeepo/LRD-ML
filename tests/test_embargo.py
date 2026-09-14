"""Tests for the point-in-time training embargo (finding #1 in docs/FINDINGS.md).

Run with:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from modules.forecast_io import n_train_rows, row_positions


def test_n_train_rows_basic_stride5():
    pos = np.arange(0, 100, 5)            # rows every 5 trading days
    t = 10                                # origin at pos 50
    # h=1: row 9 (pos 45) ends at 46 <= 50 -> all 10 rows usable
    assert n_train_rows(pos, t, h=1) == 10
    # h=5: row 9 ends at 50 == origin -> still usable
    assert n_train_rows(pos, t, h=5) == 10
    # h=22: rows with pos + 22 <= 50 -> pos <= 28 -> rows 0..5 -> 6 rows
    assert n_train_rows(pos, t, h=22) == 6


def test_no_training_target_extends_past_origin():
    rng = np.random.default_rng(0)
    # irregular spacing (after NaN-dropping rows are not evenly spaced)
    pos = np.sort(rng.choice(np.arange(0, 2000), size=300, replace=False))
    for h in (1, 5, 22, 44):
        for t in range(50, 300):
            n = n_train_rows(pos, t, h)
            assert n <= t
            if n > 0:
                assert pos[n - 1] + h <= pos[t]
            if n < t:
                assert pos[n] + h > pos[t]


def test_row_positions_maps_dates_to_trading_index():
    full = pd.bdate_range("2020-01-01", periods=60)
    rows = full[[3, 8, 13, 40]]
    assert list(row_positions(rows, full)) == [3, 8, 13, 40]
    with pytest.raises(ValueError):
        row_positions(pd.DatetimeIndex(["2019-01-01"]), full)
