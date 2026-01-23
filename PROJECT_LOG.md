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
    │   └── table4_features.tex
    ├── figures/
    │   ├── fig1_data_overview.pdf
    │   └── fig2_lrd_estimates.pdf
    └── intermediate/
        ├── vol_gph.csv
        ├── cs_dispersion.csv
        ├── rolling_d_hat.csv
        ├── cross_sectional_features.csv
        └── market_features.csv
```

---

*Last updated: 2025-01-21*
