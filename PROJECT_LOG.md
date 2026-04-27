# LRD-ML Research Project Log

**Project:** Long-Range Dependence as Informative Features for Machine Learning in Financial Forecasting
**Authors:** Akash Deep, Nicholas Appiah
**Advisor:** Dr. Svetlozar Rachev
**Institution:** Texas Tech University
**Started:** January 2025

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Data Requirements (Original Plan)](#2-data-requirements-original-plan)
3. [Data Acquisition Progress](#3-data-acquisition-progress)
4. [Implementation Progress](#4-implementation-progress)
5. [Next Steps](#5-next-steps)
6. [Session Log](#6-session-log)

---

## 1. Research Overview

### Core Hypothesis
Use **sufficient statistics** and **estimated parameters** from Long-Range Dependence (LRD) models as structured input features for ML algorithms, rather than treating LRD models and ML as competing approaches.

### Primary Research Questions
1. Does the estimated fractional differencing parameter $\hat{d}$ contain predictive information for volatility and returns beyond its role in stationarity transformation?
2. Can ML exploit time-variation in long-memory parameters to improve forecasting during regime changes?
3. What residual structure remains after ARFIMA-FIGARCH filtering, and can ML learn it?

### Key Novelties
- **$\hat{d}$ as a feature, not just a transformation:** First systematic study using memory parameters as ML inputs
- **Memory dynamics:** $\Delta\hat{d}$, Vol($\hat{d}$), Trend($\hat{d}$) as regime indicators
- **Cross-sectional memory:** Distribution of $\hat{d}$ across assets as market-level features
- **Residual structure learning:** ML-based specification testing for FIGARCH models

### Research Hypotheses
1. **H1 (Memory Parameter Predictability):** $\hat{d}$ contains predictive information for future volatility beyond HAR benchmark
2. **H2 (Memory Dynamics Signal Regimes):** Changes in $\hat{d}$ predict volatility regime transitions
3. **H3 (Cross-Sectional Memory Dispersion):** Cross-sectional dispersion in memory parameters predicts market-level volatility
4. **H4 (Residual Structure Exists):** ARFIMA-FIGARCH residuals contain learnable structure
5. **H5 (Hybrid Dominance):** LRD-augmented ML outperforms both pure econometric and pure ML approaches

---

## 2. Data Requirements (Original Plan)

### Primary Dataset: S&P 500 Constituents

#### Stock-Level Data (from Bloomberg)
| Data Type | Bloomberg Field | Frequency | Period |
|-----------|-----------------|-----------|--------|
| Adjusted Close | PX_LAST | Daily | 2000-2024 |
| Intraday Prices | INTRADAY_PRICE | 5-min | 2010-2024 |
| Volume | VOLUME | Daily | 2000-2024 |
| Bid-Ask Spread | BID_ASK_SPREAD | Daily | 2000-2024 |
| Market Cap | CUR_MKT_CAP | Monthly | 2000-2024 |
| Sector | GICS_SECTOR | Static | Current |

#### Market-Level Variables
| Variable | Bloomberg Ticker | Frequency |
|----------|------------------|-----------|
| VIX | VIX Index | Daily |
| Risk-Free Rate | USGG3M Index | Daily |
| Term Spread | USYC1030 Index | Daily |
| Credit Spread | CDX IG CDSI GEN 5Y | Daily |
| Market Return | SPX Index | Daily |

#### Sample Construction Plan
- **Training:** 2000-2017 (18 years)
- **Validation:** 2018-2019 (2 years)
- **Test:** 2020-2024 (5 years, includes COVID crisis)
- **Cross-sectional sample:** ~500 stocks with sufficient history

---

## 3. Data Acquisition Progress

### Session: January 21, 2025

#### Initial Data Available
| File | Description | Status |
|------|-------------|--------|
| `LRD ML DATA(PX_LAST).csv` | 30 stocks, daily close, 2000-2025 | Provided by user |

#### Data Downloaded (Free Sources)

**Script created:** `download_free_data.py`
**Method:** Yahoo Finance (yfinance 1.0) + FRED via pandas_datareader

| File | Content | Rows | Columns | Period | Source |
|------|---------|------|---------|--------|--------|
| `LRD ML DATA(PX_LAST).csv` | Original 30 stocks (daily close) | 6,291 | 30 | 2000-2025 | Bloomberg (provided) |
| `additional_stocks.csv` | Additional S&P 500 stocks | 6,297 | 107 | 2000-2025 | Yahoo Finance |
| `vix_data.csv` | VIX Index | 6,297 | 1 | 2000-2025 | Yahoo Finance |
| `market_indices.csv` | SPX, DJIA, NASDAQ, Russell 2000 | 6,297 | 4 | 2000-2025 | Yahoo Finance |
| `macro_data.csv` | Rates, credit spreads, oil, EUR/USD | 6,617 | 8 | 2000-2025 | FRED |
| `volume_data.csv` | Trading volume for 30 stocks | 6,297 | 30 | 2000-2025 | Yahoo Finance |
| `commodities_fx.csv` | Gold, Oil, EUR/USD | 6,326 | 3 | 2000-2025 | Yahoo Finance |

#### Macro Data Details (macro_data.csv)
| Column | Description | Source |
|--------|-------------|--------|
| Risk_Free_3M | 3-Month Treasury Bill Rate | FRED: DGS3MO |
| Treasury_10Y | 10-Year Treasury Rate | FRED: DGS10 |
| Treasury_2Y | 2-Year Treasury Rate | FRED: DGS2 |
| Term_Spread | 10Y-2Y Spread | FRED: T10Y2Y |
| IG_Credit_Spread | Investment Grade Credit Spread | FRED: BAMLC0A0CM |
| HY_Credit_Spread | High Yield Credit Spread | FRED: BAMLH0A0HYM2 |
| Oil_WTI | WTI Crude Oil Price | FRED: DCOILWTICO |
| EUR_USD | EUR/USD Exchange Rate | FRED: DEXUSEU |

#### Stock Tickers in Dataset

**Original 30 stocks (from Bloomberg):**
```
AAPL, MSFT, NVDA, JPM, BAC, GS, JNJ, PFE, UNH, AMZN,
TSLA, HD, PG, KO, PEP, XOM, CVX, COP, BA, CAT,
UNP, NEE, DUK, DOW, FCX, GOOGL, META, AMT, PLD, V
```

**Additional 107 stocks (from Yahoo Finance):**
```
Technology: IBM, INTC, ORCL, CSCO, ADBE, CRM, AVGO, TXN, QCOM, AMD
Finance: WFC, C, MS, AXP, BLK, SCHW, USB, PNC, TFC, COF
Healthcare: MRK, ABBV, LLY, TMO, ABT, DHR, BMY, AMGN, GILD, CVS
Consumer Disc: MCD, NKE, SBUX, TJX, LOW, TGT, BKNG, MAR, GM, F
Consumer Staples: WMT, COST, PM, MO, CL, KMB, GIS, K, SYY, ADM
Energy: SLB, EOG, PSX, VLO, MPC, OXY, HAL, KMI, WMB
Industrials: HON, GE, RTX, LMT, MMM, DE, EMR, ITW, ETN, FDX
Utilities: SO, D, AEP, EXC, SRE, XEL, PEG, ED, WEC, ES
Materials: LIN, APD, SHW, ECL, NEM, NUE, VMC, MLM, DD, PPG
Real Estate: SPG, EQIX, PSA, O, WELL, AVB, EQR, DLR, VTR, BXP
Communication: DIS, CMCSA, NFLX, T, VZ, TMUS, CHTR, EA
```

**Total: 137 unique stocks**

### Data Still Required (Bloomberg Terminal)

| Data | Priority | Purpose | Bloomberg Function |
|------|----------|---------|-------------------|
| **5-min intraday prices** | HIGH | Realized Volatility calculation | INTRADAY_BAR |
| High/Low daily prices | MEDIUM | Parkinson volatility estimator | PX_HIGH, PX_LOW |
| Bid-Ask spread | LOW | Liquidity features | BID_ASK_SPREAD |

#### Bloomberg Export Instructions
```
For 5-minute intraday data:
1. Open Bloomberg Terminal
2. Use Excel API or BDH function
3. Request: INTRADAY_BAR for each ticker
4. Fields: TRADE (or BID/ASK midpoint)
5. Period: 2010-01-01 to 2025-01-15
6. Export as CSV: Date, Time, Open, High, Low, Close, Volume
```

---

## 4. Implementation Progress

### Completed
- [x] Initial data assessment
- [x] Free data download script (`download_free_data.py`)
- [x] Downloaded 137 stocks daily prices (2000-2025)
- [x] Downloaded VIX (2000-2025)
- [x] Downloaded market indices (SPX, DJIA, NASDAQ, RUT)
- [x] Downloaded macro variables from FRED
- [x] Downloaded volume data
- [x] Data preprocessing pipeline (`preprocess_data.py`)
- [x] Computed log returns
- [x] Winsorized extreme returns (0.1%-99.9%)
- [x] Computed volatility proxy (squared returns)
- [x] Aligned all datasets (6,296 common trading days)
- [x] Created processed data files

### In Progress
- [ ] LRD estimation pipeline (GPH, Local Whittle estimators)
- [ ] Feature engineering framework
- [ ] Benchmark model implementation (HAR-RV, FIGARCH)

### Not Started
- [ ] Obtain 5-min intraday data from Bloomberg
- [ ] Realized Volatility calculation
- [ ] ML model development (XGBoost, LSTM)
- [ ] Cross-sectional memory analysis
- [ ] Residual learning framework
- [ ] Statistical evaluation (DM tests, MCS)

---

## 5. Next Steps

### Immediate (Can do now with current data)
1. **Build LRD estimation pipeline**
   - Implement GPH estimator
   - Implement Local Whittle estimator
   - Rolling window estimation for $\hat{d}$

2. **Create volatility proxy from daily data**
   - Squared returns: $\sigma_t^2 \approx r_t^2$
   - Parkinson estimator (if High/Low available)

3. **Implement benchmark models**
   - HAR-RV model
   - FIGARCH(1,d,1) model

### After Bloomberg Data
4. **Compute Realized Volatility from 5-min data**
   - Standard RV: $RV_t = \sum_{i=1}^{M} r_{t,i}^2$
   - Bias-corrected RV with autocorrelation adjustment

5. **Full feature engineering**
   - Memory dynamics features
   - Cross-sectional features

6. **ML model training and evaluation**

---

## 6. Session Log

### 2025-01-21: Project Initialization

**Actions taken:**
1. Reviewed research proposal (LaTeX document)
2. Assessed available data (`LRD ML DATA(PX_LAST).csv`)
3. Identified data gaps (5-min intraday, macro variables, more stocks)
4. Created `download_free_data.py` script
5. Downloaded free data from Yahoo Finance and FRED
6. Upgraded yfinance to v1.0 (fixed API issues)
7. Created this project log

**Issues encountered:**
- yfinance 0.2.28 had API issues ("No timezone found" errors)
- Fixed by upgrading to yfinance 1.0
- yfinance 1.0 returns MultiIndex columns - handled in code

**Data summary:**
- 137 stocks with 25 years of daily data
- VIX and major indices available
- Macro variables (rates, spreads) from FRED
- Still need: 5-min intraday data from Bloomberg

**Next session priorities:**
1. Start LRD estimation pipeline
2. Implement GPH and Local Whittle estimators
3. Begin with daily volatility proxy while awaiting Bloomberg data

---

### 2025-01-21 (continued): Data Preprocessing

**Actions taken:**
1. Created `preprocess_data.py` preprocessing pipeline
2. Identified data quality issues:
   - Original data in reverse chronological order (fixed)
   - Missing values for newer stocks (TSLA, META, DOW, V - pre-IPO)
   - 3 extreme returns >50% (stock splits)
   - Macro data ~5% missing (weekends/holidays)
3. Applied preprocessing steps:
   - Reversed date order
   - Combined original + additional stocks (137 total)
   - Filtered by 70% coverage (125 stocks retained)
   - Computed log returns
   - Winsorized returns at 0.1% and 99.9% percentiles (clipped 0.2% of values)
   - Computed volatility proxy (squared returns)
   - Aligned all datasets to 6,296 common trading days
   - Forward-filled macro data gaps

**Processed data created:**
| File | Content | Shape |
|------|---------|-------|
| `processed/returns.csv` | Log returns | (6296, 125) |
| `processed/volatility_proxy.csv` | Squared returns | (6296, 125) |
| `processed/prices.csv` | Cleaned prices | (6296, 125) |
| `processed/market.csv` | VIX + indices + returns | (6296, 9) |
| `processed/macro.csv` | Macro variables | (6296, 8) |
| `processed/summary_stats.csv` | Stock statistics | (125, 10) |

**Data quality after preprocessing:**
- 125 stocks with 70%+ coverage
- 6,296 trading days (2000-01-04 to 2025-01-14)
- Missing values: 0.79% in returns (pre-IPO dates)
- Mean annualized return: 9.74%
- Mean annualized volatility: 31.57%

**Stocks removed (insufficient data):**
TSLA, DOW, META, V, AVGO, ABBV, TMUS, CRM, GM, PSX, MPC, KMI

---

### 2025-01-21 (continued): Module 1 - Data Description

**Actions taken:**
1. Created modular work plan (`WORK_PLAN.md`)
2. Set up directory structure (`modules/`, `results/tables/`, `results/figures/`)
3. Implemented `module1_data_description.py`
4. Generated paper outputs

**Paper outputs created:**
| Output | File | Description |
|--------|------|-------------|
| **Table 1** | `results/tables/table1_summary_stats.tex` | Summary statistics by sector |
| **Table 2** | `results/tables/table2_data_description.tex` | Data description & macro vars |
| **Figure 1** | `results/figures/fig1_data_overview.pdf` | Returns, ACF, VIX visualization |

**Key findings from Module 1:**
- Mean annualized return: 9.67%
- Mean annualized volatility: 32.47%
- Excess kurtosis: 7.98 (fat tails confirmed)
- Returns ACF ≈ 0, Squared returns ACF significant → LRD evidence
- VIX range: 9.14 to 82.69 (COVID peak)

**Sector breakdown:**
- Highest volatility: Technology (38.73%), Financials (36.70%)
- Lowest volatility: Consumer Staples (22.83%), Utilities (22.93%)
- Highest kurtosis: Utilities (10.64) - more extreme events

---

### 2025-01-21 (continued): Module 2 - LRD Estimation

**Actions taken:**
1. Implemented GPH (Geweke-Porter-Hudak) estimator
2. Implemented Local Whittle estimator (corrected optimization)
3. Computed cross-sectional LRD estimates for all 125 stocks
4. Computed rolling window estimates (500-day window)
5. Computed cross-sectional memory dispersion over time
6. Generated paper outputs

**Paper outputs created:**
| Output | File | Description |
|--------|------|-------------|
| **Table 3** | `results/tables/table3_lrd_estimates.tex` | LRD estimates by sector |
| **Figure 2** | `results/figures/fig2_lrd_estimates.pdf` | Distribution, rolling, dispersion |

**Key findings from Module 2:**
| Series | Estimator | Mean d | % Significant |
|--------|-----------|--------|---------------|
| Returns | GPH | -0.015 | 0.8% |
| Returns | Local Whittle | -0.034 | 32.0% |
| **Volatility** | **GPH** | **0.213** | **100%** |
| **Volatility** | **Local Whittle** | **0.413** | **100%** |

**Critical findings for the paper:**
1. Returns show NO long memory (d ≈ 0) - consistent with market efficiency
2. Volatility shows STRONG long memory (d ≈ 0.2-0.4) - 100% significant
3. Memory parameter INCREASES during crises (2008, 2020)
4. Cross-sectional mean d is POSITIVELY correlated with VIX (ρ = 0.47)
5. This supports Hypothesis H2: memory dynamics signal regimes

---

## File Structure

```
LRD_Nicholas/
├── PROJECT_LOG.md              # This documentation file
├── download_free_data.py       # Data download script
├── preprocess_data.py          # Data preprocessing pipeline
│
├── [Raw Data]
│   ├── LRD ML DATA(PX_LAST).csv    # Original 30 stocks (Bloomberg)
│   ├── additional_stocks.csv       # 107 additional stocks (Yahoo)
│   ├── vix_data.csv               # VIX index
│   ├── market_indices.csv         # SPX, DJIA, NASDAQ, RUT
│   ├── macro_data.csv             # FRED macro variables
│   ├── volume_data.csv            # Trading volume
│   └── commodities_fx.csv         # Gold, Oil, EUR/USD
│
├── processed/                  # Cleaned & aligned data (use these!)
│   ├── returns.csv            # Log returns (6296 x 125)
│   ├── volatility_proxy.csv   # Squared returns (6296 x 125)
│   ├── prices.csv             # Clean prices (6296 x 125)
│   ├── market.csv             # VIX + indices + returns (6296 x 9)
│   ├── macro.csv              # Macro variables (6296 x 8)
│   └── summary_stats.csv      # Per-stock statistics (125 x 10)
│
└── [To Be Created]
    ├── lrd_estimation.py      # GPH, Local Whittle estimators
    ├── feature_engineering.py # LRD feature construction
    ├── models/                # HAR, FIGARCH, ML models
    └── results/               # Output and figures
```

---

### 2025-01-21 (continued): Module 3 - Feature Engineering

**Actions taken:**
1. Implemented comprehensive feature engineering pipeline (`module3_feature_engineering.py`)
2. Computed HAR components (RV_d, RV_w, RV_m) for all 125 stocks
3. Computed rolling LRD estimates (500-day window, every 5 days for efficiency)
4. Computed memory dynamics features (delta_d, vol_d, trend_d)
5. Computed cross-sectional memory features (cs_mean_d, cs_std_d, cs_skew_d)
6. Created market features from VIX and macro variables
7. Generated paper outputs

**Paper outputs created:**
| Output | File | Description |
|--------|------|-------------|
| **Table 4** | `results/tables/table4_features.tex` | Feature definitions & statistics |

**Intermediate outputs saved:**
| File | Description | Shape |
|------|-------------|-------|
| `rolling_d_hat.csv` | Rolling GPH d estimates | (5796, 125) |
| `cross_sectional_features.csv` | Market-level memory features | (5796, 7) |
| `market_features.csv` | VIX + transformed macro | (6296, 25) |

**Feature categories created:**

| Category | Features | Description |
|----------|----------|-------------|
| **HAR** | RV_d, RV_w, RV_m | Daily, weekly, monthly realized volatility |
| **LRD** | d_hat, se_d, d_significant | Rolling memory parameter estimates |
| **Memory Dynamics** | delta_d, vol_d, trend_d | Changes, volatility, trend in d |
| **Cross-Sectional** | cs_mean_d, cs_std_d, cs_skew_d | Market-level memory distribution |
| **Market** | VIX, spreads, rates | Macro environment features |

**Key statistics:**
- Rolling d_hat: mean=0.096, std=0.075, range=[-0.07, 0.27]
- Cross-sectional mean d: mean=0.108, std=0.072
- Cross-sectional dispersion: std=0.060 (relatively stable)
- VIX: mean=19.87, range=[9.14, 82.69]

---

## File Structure

```
LRD_Nicholas/
├── PROJECT_LOG.md              # This documentation file
├── WORK_PLAN.md                # Modular work plan
├── download_free_data.py       # Data download script
├── preprocess_data.py          # Data preprocessing pipeline
│
├── modules/
│   ├── module1_data_description.py   # Table 1-2, Figure 1
│   ├── module2_lrd_estimation.py     # Table 3, Figure 2
│   └── module3_feature_engineering.py # Table 4
│
├── [Raw Data]
│   ├── LRD ML DATA(PX_LAST).csv    # Original 30 stocks (Bloomberg)
│   ├── additional_stocks.csv       # 107 additional stocks (Yahoo)
│   ├── vix_data.csv               # VIX index
│   ├── market_indices.csv         # SPX, DJIA, NASDAQ, RUT
│   ├── macro_data.csv             # FRED macro variables
│   ├── volume_data.csv            # Trading volume
│   └── commodities_fx.csv         # Gold, Oil, EUR/USD
│
├── processed/                  # Cleaned & aligned data
│   ├── returns.csv            # Log returns (6296 x 125)
│   ├── volatility_proxy.csv   # Squared returns (6296 x 125)
│   ├── prices.csv             # Clean prices (6296 x 125)
│   ├── market.csv             # VIX + indices + returns (6296 x 9)
│   ├── macro.csv              # Macro variables (6296 x 8)
│   └── summary_stats.csv      # Per-stock statistics (125 x 10)
│
└── results/
    ├── tables/
    │   ├── table1_summary_stats.tex
    │   ├── table2_data_description.tex
    │   ├── table3_lrd_estimates.tex
    │   ├── table4_features.tex
    │   └── table_figarch_estimates.tex
    ├── figures/
    │   ├── fig1_data_overview.pdf
    │   ├── fig2_lrd_estimates.pdf
    │   └── fig_preliminary_diagnostics.pdf
    └── intermediate/
        ├── vol_gph.csv
        ├── cs_dispersion.csv
        ├── rolling_d_hat.csv
        ├── cross_sectional_features.csv
        └── market_features.csv
```

---

## Session: 2025-01-30 - Preliminary Diagnostics and FIGARCH

### Objective
Add preliminary diagnostics section with:
1. Log-return distribution analysis
2. Cross-stock volatility correlation heatmap
3. ARIMA-FIGARCH fitted volatility
4. QQ-plot for normality assessment

### Implemented
- **New module:** `modules/module1b_preliminary_diagnostics.py`
  - FIGARCH(1,d,1) model fitting using `arch` library
  - Diagnostic figure with 4 panels
  - FIGARCH comparison table

### Key Results
1. **Return Distribution:**
   - Excess kurtosis: 7.98
   - Jarque-Bera: 2,070,028 (p < 0.001)
   - Student-t with ~2.6 df fits better than normal

2. **FIGARCH Estimates:**
   - Mean d = 0.329 (range: 0.235 to 0.461)
   - Consistent with semi-parametric estimates (GPH: 0.213, LW: 0.413)
   - Validates long memory across all 10 sample stocks

3. **Cross-Stock Correlations:**
   - Volatility correlations range from 0.20 to 0.48
   - Supports use of cross-sectional features

### New Outputs
- `results/figures/fig_preliminary_diagnostics.pdf`
- `results/figures/fig_preliminary_diagnostics.png`
- `results/tables/table_figarch_estimates.tex`

### Paper Updates
- Added "Preliminary Diagnostics" subsection to Section 3 (Data)
- Added FIGARCH methodology to Section 2 (Methodology)
- Added Figure 2: Preliminary Diagnostics
- Added Table: FIGARCH(1,d,1) Model Estimates
- Added Baillie et al. (1996) reference

---

## Session: 2026-04-25 — Bloomberg pull, panel rebuild, and Phases 1–3

### Context
Following Prof. Rachev's 19 April 2026 feedback (the extended framework
*Memory, Roughness, and Information Persistence in Financial Markets*), we
restructured the data, code, and outputs around the new 10-section paper plan
(see [REVISION_PLAN.tex](REVISION_PLAN.tex)).

### Bloomberg pull and clean-up
Nicholas pulled five files into `bloomberg_pull/`:
`OHLCV.xlsx`, `intraday.xlsx`, `sector_ETF.xlsx`, `market_level_daily.csv`,
`stock_metadata.csv`.

Issues found and fixed by `preprocess_bloomberg.py` and
`preprocess_supporting.py`:

| Issue | Fix |
|---|---|
| `AAPL` sheet contained AAP (Advance Auto Parts) data — closes matched to the cent ($58, $3.5B mcap; not Apple) | Replaced with real AAPL pulled from yfinance for the same window |
| Sheet named `JMP` (data was correct JPM) | Renamed → `JPM` |
| `TXN` Bloomberg sheet was empty | Pulled from yfinance |
| Sector ETF file in Bloomberg multi-block wide layout | Flattened into one panel per OHLCV field |
| Market-level CSV had `"Last Px"` header noise row | Stripped, columns renamed (`SPX`, `INDU`, `NDX`, `RTY`, `VIX`, `MOVE`, `USGG3M`, `USGG10YR`, `USYC2Y10`, `CDX_IG_5Y`, `CDX_HY_5Y`) |
| Stock metadata file used company names as join key | Mapped to ticker order from `OHLCV.xlsx`; added `SizeBucket` (Small/Mid/Large) by mcap terciles |

Constraints accepted:
- **Intraday** only Oct 2025 → Apr 2026 (Terminal 1-year cap). Will be used as a
  recent-period high-frequency robustness slice, not for rolling estimates.
- **Daily history** starts 2001-11-29 (Bloomberg's earliest for some tickers).
- **CDX IG/HY** start 2011-09-09; CDX HY column is a *price* index (Bloomberg
  field `CDX HY CDSI GEN 5Y PRC Corp`), not spread.
- **AAPL/TXN historical mcap** not available from yfinance; static current mcap
  in `metadata.csv` is sufficient for size-bucket assignment.

Output panels under `bloomberg_pull/processed/`:
- `prices_close.csv`, `prices_open.csv`, `prices_high.csv`, `prices_low.csv` (all 6,136 × 125)
- `volume.csv`, `mktcap.csv`
- `ohlcv_long.csv` (long format, 732,892 × 7)
- `metadata.csv` (125 × 6)
- `market_level.csv` (6,322 × 11)
- `sector_etf_{open,close,high,low,volume}.csv` (6,538 × 11 each)
- `rv_parkinson.csv` (6,136 × 125) — primary forecasting target

### Phase 1 — Repoint pipeline to new data
- New module `modules/io_v2.py` — single source of truth for loading +
  70%-coverage filter + winsorized log returns + log RV + market alignment.
  Caches under `bloomberg_pull/processed/clean_panel/`.
- Filter retained **115 / 125** stocks. Dropped (insufficient history): ABBV,
  AVGO, DG, DOW, HCA, HLT, KHC, META, TSLA, ZTS.
- `module1_data_description.py` rewritten — uses real GICS sectors (11),
  Parkinson RV in Figure 1 panel (b), three-way ACF comparison in panel (c)
  (returns vs squared returns vs Parkinson RV).
- `module1b_preliminary_diagnostics.py` rewritten — same 4-panel diagnostic
  figure with Parkinson sqrt(RV) in panel (c), all 10 sample-stock FIGARCH
  fits succeeded (no GARCH fallback this time).

Headline numbers (refreshed):
- Pooled return: mean 9.38% ann, std 31.48% ann, excess kurtosis 11.64,
  JB ≈ 3.95M
- VIX: 9.14 → 82.69 (covers GFC + COVID peaks)
- FIGARCH(1,d,1): mean $\hat d$ = **0.329** across 10 stocks, range
  [0.237, 0.461]

Outputs refreshed: `table1_summary_stats.tex`, `table2_data_description.tex`,
`table_figarch_estimates.tex`, `fig1_data_overview.pdf`,
`fig_preliminary_diagnostics.pdf`.

### Phase 2 — Estimation layer (LRD + roughness)
`module2_lrd_estimation.py` rewritten end-to-end:
- Static cross-sectional GPH and Local Whittle on returns and Parkinson RV.
- New rolling Hurst / roughness estimator via the scaling method on
  log Parkinson RV (qth-moment increments over lags {1,2,3,5,8,13,21}).
- Rolling weekly panels (window 750, stride 5) for d_GPH, d_LW, H — feeds
  Module 3 directly.

| Series | Estimator | Mean d | % Significant |
|---|---|---|---|
| Returns | GPH | −0.011 | 0% |
| Returns | LW | −0.028 | 19% |
| Parkinson RV | GPH | **+0.226** | **98%** |
| Parkinson RV | LW | **+0.440** | **100%** |
| log(RV) | Hurst | **H = 0.063** | 100% have H<0.5 |

**Key result for the paper.** Cross-sectional mean $\hat d_t$ by regime:
| Period | Mean $\hat d_t$ | vs calm |
|---|---|---|
| Calm 2013–2014 | +0.154 | baseline |
| GFC 2008-Q3 → 2009-Q4 | **+0.259** | +68% |
| COVID 2020 | **+0.287** | +86% |

Correlation $\rho(\text{VIX}, \text{mean } \hat d_t)$ = **+0.501**.

Roughness exponent H ≈ 0.06 across all 115 stocks places our panel firmly in
the rough-volatility range of Gatheral–Jaisson–Rosenbaum (2018).

Outputs: `table3_lrd_estimates.tex`, `fig2_lrd_estimates.pdf`,
intermediate panels `lrd_*.csv`, `hurst_rv_log.csv`,
`rolling_d_gph.csv`, `rolling_d_lw.csv`, `rolling_hurst.csv`
(each rolling panel is 1078 × 115).

### Phase 3 — Persistence feature vector
`module3_feature_engineering.py` rewritten to consume Phase 2 outputs and
produce the full Z_t vector per Rachev's framework Section 6:

23 panels saved under `results/intermediate/features/`:
- LRD: `feat_d_gph`, `feat_d_lw`, `feat_h`
- Memory dynamics: `feat_delta_d_{gph,lw}`, `feat_vol_d_{gph,lw}`,
  `feat_trend_d_{gph,lw}`, `feat_delta_h`
- HAR: `feat_har_{d,w,m}` (built from Parkinson RV, non-anticipative)
- Sector aggregate: `feat_sector_mean_d` (each stock gets its own
  GICS-sector mean)
- Cross-sectional: `feat_cross_section.csv` (mean, std, median, skew, kurt,
  range, $P(\hat d > 0.30)$ over time)
- Threshold flags: `feat_threshold_{10,20,30,40}`
- Interactions: `feat_d_x_{vix,move,illiq}` (illiquidity = inverse of 22-day
  rolling dollar volume)
- Market axis: `market_axes.csv` (VIX, MOVE, USYC2Y10, USGG10YR, CDX IG/HY)

Sanity check: AAPL `d × VIX` mean = 1.05 in calm 2013–14, **12.16 in GFC**,
**8.22 in COVID** — interaction term amplifies crisis predictability exactly
as Rachev's framework predicts.

Outputs: `table4_features.tex` (categorized feature definitions and pooled
statistics for all categories: LRD, Roughness, Memory dynamics, HAR, Sector,
Cross-sectional, Market, Interaction).

### Cross-phase consistency (verified end-to-end)
- All Phase 1 panels: shape (6136, 115); all Phase 2/3 rolling panels: (1078, 115)
- Tickers identical across Phase 1, 2, 3
- Phase 2 `rolling_d_gph` ≡ Phase 3 `feat_d_gph` (zero diff)
- Interaction `d × VIX` reconstructs to within 4e-15 of `d * VIX`
- HAR_d at sample date $t$ equals raw Parkinson RV on the prior trading day —
  confirms HAR features are non-anticipative

### What's next
Phase 4: build `module4_benchmarks.py` (extended: AR, HAR, ARFIMA, FIGARCH on
log RV), `module5_ml_models.py` (nested A/B/C/D — Lasso/Ridge/EN, RF, GBM at
horizons h ∈ {1, 5, 22}), `module6_forecast_eval.py` (MSE + QLIKE,
Diebold–Mariano tests, regime/sector/size splits).

---

*Last updated: 2026-04-25*
