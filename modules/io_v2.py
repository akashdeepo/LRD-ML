"""
io_v2.py — single source of truth for loading the new Bloomberg-based panel.

All downstream modules (Module 1 onwards) should import from here so that the
70% coverage filter, log-return construction, and target/feature alignment
are computed once, consistently.

Inputs (under bloomberg_pull/processed/):
  prices_close.csv, prices_open.csv, prices_high.csv, prices_low.csv,
  volume.csv, mktcap.csv, rv_parkinson.csv, metadata.csv,
  market_level.csv, sector_etf_close.csv (and other ETF fields).

Outputs (cached under bloomberg_pull/processed/clean_panel/):
  returns.csv, log_rv.csv, prices_close.csv, prices_high.csv, prices_low.csv,
  rv_parkinson.csv, market.csv, metadata.csv, kept_tickers.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "bloomberg_pull" / "processed"
CLEAN = RAW / "clean_panel"

COVERAGE_THRESHOLD = 0.70
WINSOR_LOWER, WINSOR_UPPER = 0.001, 0.999


@dataclass
class Panel:
    prices: pd.DataFrame      # close prices, T x N
    high: pd.DataFrame        # daily high
    low: pd.DataFrame         # daily low
    returns: pd.DataFrame     # winsorized log returns
    rv_parkinson: pd.DataFrame  # range-based RV (variance scale)
    log_rv: pd.DataFrame      # log of (rv_parkinson + tiny floor)
    market: pd.DataFrame      # market_level columns + VIX
    metadata: pd.DataFrame    # ticker -> sector, size bucket, etc.
    kept: list[str]           # tickers retained after coverage filter


def _winsorize(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    qlo = df.quantile(lower)
    qhi = df.quantile(upper)
    return df.clip(lower=qlo, upper=qhi, axis=1)


def build_clean_panel(force: bool = False) -> Panel:
    if (not force) and CLEAN.exists() and (CLEAN / "kept_tickers.txt").exists():
        return load_clean_panel()

    CLEAN.mkdir(exist_ok=True)

    prices = pd.read_csv(RAW / "prices_close.csv", index_col=0, parse_dates=True)
    high = pd.read_csv(RAW / "prices_high.csv", index_col=0, parse_dates=True)
    low = pd.read_csv(RAW / "prices_low.csv", index_col=0, parse_dates=True)
    rv_pk = pd.read_csv(RAW / "rv_parkinson.csv", index_col=0, parse_dates=True)
    mkt = pd.read_csv(RAW / "market_level.csv", index_col=0, parse_dates=True)
    meta = pd.read_csv(RAW / "metadata.csv")

    # Coverage filter
    coverage = prices.notna().mean()
    kept = sorted(coverage[coverage >= COVERAGE_THRESHOLD].index.tolist())
    dropped = sorted(set(prices.columns) - set(kept))
    print(f"  coverage filter: kept {len(kept)} / {prices.shape[1]} stocks "
          f"(dropped {len(dropped)}: {dropped[:10]}{'...' if len(dropped) > 10 else ''})")

    prices = prices[kept]
    high = high[kept]
    low = low[kept]
    rv_pk = rv_pk[kept]

    # Log returns + winsorize
    returns = np.log(prices / prices.shift(1))
    returns = _winsorize(returns, WINSOR_LOWER, WINSOR_UPPER)

    # Log RV (add a tiny floor to avoid log(0))
    floor = max(rv_pk.replace(0, np.nan).min().min() / 10.0, 1e-12)
    log_rv = np.log(rv_pk.where(rv_pk > 0, floor))

    # Align market data to the price calendar; forward-fill modest gaps
    market = mkt.reindex(prices.index).ffill(limit=3)

    # Metadata: restrict to kept tickers
    meta = meta.set_index("BloombergTicker").loc[kept].reset_index()

    # Persist
    prices.to_csv(CLEAN / "prices_close.csv")
    high.to_csv(CLEAN / "prices_high.csv")
    low.to_csv(CLEAN / "prices_low.csv")
    returns.to_csv(CLEAN / "returns.csv")
    rv_pk.to_csv(CLEAN / "rv_parkinson.csv")
    log_rv.to_csv(CLEAN / "log_rv.csv")
    market.to_csv(CLEAN / "market.csv")
    meta.to_csv(CLEAN / "metadata.csv", index=False)
    (CLEAN / "kept_tickers.txt").write_text("\n".join(kept))

    return Panel(prices, high, low, returns, rv_pk, log_rv, market, meta, kept)


def load_clean_panel() -> Panel:
    prices = pd.read_csv(CLEAN / "prices_close.csv", index_col=0, parse_dates=True)
    high = pd.read_csv(CLEAN / "prices_high.csv", index_col=0, parse_dates=True)
    low = pd.read_csv(CLEAN / "prices_low.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(CLEAN / "returns.csv", index_col=0, parse_dates=True)
    rv_pk = pd.read_csv(CLEAN / "rv_parkinson.csv", index_col=0, parse_dates=True)
    log_rv = pd.read_csv(CLEAN / "log_rv.csv", index_col=0, parse_dates=True)
    market = pd.read_csv(CLEAN / "market.csv", index_col=0, parse_dates=True)
    meta = pd.read_csv(CLEAN / "metadata.csv")
    kept = (CLEAN / "kept_tickers.txt").read_text().splitlines()
    return Panel(prices, high, low, returns, rv_pk, log_rv, market, meta, kept)


def sector_map(meta: pd.DataFrame) -> dict:
    return dict(zip(meta["BloombergTicker"], meta["GICS_Sector"]))


if __name__ == "__main__":
    p = build_clean_panel(force=True)
    print(f"\nClean panel built:")
    print(f"  prices       {p.prices.shape}  range {p.prices.index.min().date()} -> {p.prices.index.max().date()}")
    print(f"  returns      {p.returns.shape}")
    print(f"  rv_parkinson {p.rv_parkinson.shape}")
    print(f"  market       {p.market.shape}  cols {list(p.market.columns)}")
    print(f"  metadata     {p.metadata.shape}")
    print(f"  kept tickers {len(p.kept)}")
