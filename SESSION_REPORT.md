# Session Report — 2026-04-25

**Project:** Long-Range Dependence as Informative Features for ML Volatility Forecasting (revised per Rachev 19 April 2026 feedback)

**Authors:** Akash Deep · Nicholas Appiah · Prof. Svetlozar T. Rachev (advisor / co-author)

This document records the current state of the project after the Bloomberg
data refresh and Phases 1–3 of the [REVISION_PLAN.tex](REVISION_PLAN.tex)
execution. It is meant to be read alone; it duplicates the session entry in
[PROJECT_LOG.md](PROJECT_LOG.md) with extra structure for handoff.

---

## 1. What is in place

### Data
| File | Shape | Source / status |
|---|---|---|
| `bloomberg_pull/processed/prices_close.csv` | 6136 × 125 | Bloomberg (AAPL/TXN replaced with yfinance; JMP→JPM) |
| `bloomberg_pull/processed/prices_high.csv`, `prices_low.csv`, `prices_open.csv` | 6136 × 125 each | Bloomberg, cleaned |
| `bloomberg_pull/processed/volume.csv`, `mktcap.csv` | 6136 × 125 each | Bloomberg, cleaned |
| `bloomberg_pull/processed/rv_parkinson.csv` | 6136 × 125 | Derived from H/L (range estimator) |
| `bloomberg_pull/processed/market_level.csv` | 6322 × 11 | Bloomberg (SPX, INDU, NDX, RTY, VIX, MOVE, rates, CDX) |
| `bloomberg_pull/processed/sector_etf_*.csv` | 6538 × 11 each | Bloomberg, flattened |
| `bloomberg_pull/processed/metadata.csv` | 125 × 6 | GICS sector + size buckets |
| `bloomberg_pull/intraday.xlsx` | ~10.6k rows | Oct 2025 → Apr 2026 only (Terminal cap) |

### Filtered analysis panel (`bloomberg_pull/processed/clean_panel/`)
| File | Shape | Built by |
|---|---|---|
| `prices_close.csv`, `high`, `low` | 6136 × 115 | `modules/io_v2.py` |
| `returns.csv` (winsorized log returns) | 6136 × 115 | `modules/io_v2.py` |
| `rv_parkinson.csv` | 6136 × 115 | `modules/io_v2.py` |
| `log_rv.csv` | 6136 × 115 | `modules/io_v2.py` |
| `market.csv` | 6136 × 11 | `modules/io_v2.py` |
| `metadata.csv` | 115 × 6 | `modules/io_v2.py` |
| `kept_tickers.txt` | 115 lines | `modules/io_v2.py` |

70%-coverage filter dropped: ABBV, AVGO, DG, DOW, HCA, HLT, KHC, META, TSLA, ZTS.

### LRD + roughness intermediate panels (`results/intermediate/`)
| File | Shape |
|---|---|
| `lrd_returns_gph.csv`, `lrd_returns_lw.csv` | 115 × 4 (one-shot per stock) |
| `lrd_rv_gph.csv`, `lrd_rv_lw.csv` | 115 × 4 |
| `hurst_rv_log.csv` | 115 × 1 |
| `rolling_d_gph.csv`, `rolling_d_lw.csv`, `rolling_hurst.csv` | 1078 × 115 (weekly stride, 750-day window) |

### Persistence feature vector (`results/intermediate/features/`)
23 panels:
- LRD: `feat_d_gph`, `feat_d_lw`, `feat_h`
- Memory dynamics: `feat_delta_d_{gph,lw}`, `feat_vol_d_{gph,lw}`, `feat_trend_d_{gph,lw}`, `feat_delta_h`
- HAR: `feat_har_{d,w,m}` (Parkinson RV, non-anticipative)
- Sector aggregate: `feat_sector_mean_d`
- Threshold flags: `feat_threshold_{10,20,30,40}`
- Interactions: `feat_d_x_{vix,move,illiq}`
- Cross-sectional moments: `feat_cross_section.csv` (mean, std, median, skew, kurt, range, share above 0.30)
- Market axis: `market_axes.csv`

### Paper outputs (current set)
| File | Refresh status |
|---|---|
| `results/tables/table1_summary_stats.tex` | refreshed Phase 1 |
| `results/tables/table2_data_description.tex` | refreshed Phase 1 |
| `results/tables/table3_lrd_estimates.tex` | refreshed Phase 2 (now includes Hurst column) |
| `results/tables/table4_features.tex` | refreshed Phase 3 (full feature vector by category) |
| `results/tables/table_figarch_estimates.tex` | refreshed Phase 1b |
| `results/figures/fig1_data_overview.pdf` | refreshed Phase 1 |
| `results/figures/fig2_lrd_estimates.pdf` | refreshed Phase 2 |
| `results/figures/fig_preliminary_diagnostics.pdf` | refreshed Phase 1b |
| `results/tables/table5_*.tex`, `table6_*.tex`, `table7_*.tex`, `table8_*.tex`, `fig5_interpretation.pdf` | **stale**, from January 2025; regenerate in Phase 4–5 |

---

## 2. Headline numbers

### Sample
- 115 stocks · 6,136 trading days · 2001-11-29 → 2026-04-21 · 251.6 trading days/year

### Distribution (pooled returns)
- Mean (annualized): 9.38%
- Std (annualized): 31.48%
- Excess kurtosis: 11.64
- Jarque–Bera: 3,945,618 — overwhelming non-normality
- Student-$t$ fit df: ≈ 2.7

### Long memory
| Series | Estimator | Mean d̂ | % significant at 5% |
|---|---|---|---|
| Returns | GPH | −0.011 | 0% |
| Returns | LW | −0.028 | 19% |
| Parkinson RV | GPH | **+0.226** | **98%** |
| Parkinson RV | LW | **+0.440** | **100%** |
| FIGARCH(1,d,1) on returns (10 stocks) | param | +0.329 | 100% |

### Roughness
- Hurst exponent of log Parkinson RV: mean **H = 0.063** (range 0.04–0.11)
- All 115 stocks have $H < 0.5$ — sharply rough, in line with Gatheral–Jaisson–Rosenbaum (2018)

### Regime structure (this is the result Section 8 will lead with)
| Period | Cross-sectional mean $\hat d_t$ | vs calm |
|---|---|---|
| Calm 2013–2014 | +0.154 | baseline |
| GFC 2008-Q3 → 2009-Q4 | **+0.259** | **+68%** |
| COVID 2020 | **+0.287** | **+86%** |

$\rho(\text{VIX},\ \text{mean } \hat d_t) = +0.501$.

### Crisis amplification of the persistence × VIX interaction (AAPL example)
- Calm 2013–14: `d × VIX` mean = 1.05
- GFC 2008–09: **12.16** (12× calm)
- COVID 2020: **8.22** (8× calm)

---

## 3. Code inventory

| File | Role |
|---|---|
| `preprocess_bloomberg.py` | Convert `OHLCV.xlsx` → clean per-field CSV panels; fix AAPL/JMP/TXN |
| `preprocess_supporting.py` | Clean metadata, market-level, sector ETFs; build Parkinson RV |
| `modules/io_v2.py` | Single source of truth for loading + 70% coverage filter + winsor + log returns + log RV |
| `modules/module1_data_description.py` | Tables 1–2, Figure 1 |
| `modules/module1b_preliminary_diagnostics.py` | FIGARCH table + 4-panel diagnostic figure |
| `modules/module2_lrd_estimation.py` | GPH, Local Whittle, Hurst (scaling), rolling panels, Table 3, Figure 2 |
| `modules/module3_feature_engineering.py` | Builds the full $\mathbf Z_t$ feature vector, Table 4 |
| `modules/module4_benchmarks.py` | **stale** (HAR-RV from Jan 2025). To rewrite in Phase 4 |
| `modules/module7_interpretation.py`, `module8_robustness.py` | **stale**. To rework in Phase 5 |

Run order to reproduce from scratch:
```bash
python preprocess_bloomberg.py
python preprocess_supporting.py
python -m modules.io_v2          # builds clean_panel/
python -m modules.module1_data_description
python -m modules.module1b_preliminary_diagnostics
python -m modules.module2_lrd_estimation
python -m modules.module3_feature_engineering
```

---

## 4. Cross-phase consistency (verified)

- Phase 1 panels: all (6136, 115); Phase 2/3 rolling panels: all (1078, 115).
- Tickers identical across phases.
- Phase 2 `rolling_d_gph` $\equiv$ Phase 3 `feat_d_gph` (zero diff).
- Interaction `d × VIX` reconstructs `d * VIX` to within 4e-15.
- HAR_d at sample date $t$ equals Parkinson RV on the prior trading day — non-anticipative.

---

## 5. Known caveats (carry into the paper as limitations)

1. **Survivorship**: panel is current S&P 500 constituents; some history truncated by IPO date (handled by 70% coverage filter).
2. **Daily history starts 2001-11-29** — Bloomberg's earliest for some tickers; ~22 months shorter than the original 2000–2025 plan. Sample still covers GFC, COVID, post-COVID inflation.
3. **Intraday 5-min data only Oct 2025 → Apr 2026** — Bloomberg Terminal 1-year cap. Used only as a recent-period high-frequency robustness slice; main RV proxy is Parkinson on daily H/L (full panel, full history).
4. **CDX IG and HY** start 2011-09-09; CDX HY column is a price index, not a spread.
5. **AAPL and TXN historical mcap** unavailable from yfinance — static current mcap from `metadata.csv` is used for size-bucket assignment.
6. **HAR magnitudes (~3e-4)** look small in Table 4 because Parkinson is in raw variance units — models normalize internally.

---

## 6. Next phase (Phase 4 — forecasting pipeline)

Per [REVISION_PLAN.tex](REVISION_PLAN.tex):
- `module4_benchmarks.py`: AR(p), HAR-RV, ARFIMA on log RV, FIGARCH(1,d,1) on returns (subset).
- `module5_ml_models.py`: nested Models A → B → C → D (Lasso/Ridge/EN, RF, GBM), rolling re-estimation, horizons $h \in \{1, 5, 22\}$.
- `module6_forecast_eval.py`: MSE + QLIKE, Diebold–Mariano tests, regime split (VIX quartile + GFC/COVID windows), sector/size splits.
- Output: Tables 5 (model comparison), 6 (feature importance), 7 (subsamples), 8 (horizons) — these will populate Section 8 of the paper.
