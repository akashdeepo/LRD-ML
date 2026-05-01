# Memory, Roughness, and Information Persistence in Financial Markets

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A structural and machine-learning approach to volatility forecasting using
long-memory and rough-volatility features as state-dependent indicators of
information persistence.

## Paper

> **Memory, Roughness, and Information Persistence in Financial Markets:
> A Structural and Machine Learning Approach to Volatility Forecasting**
>
> Akash Deep, Nicholas Appiah, Svetlozar T. Rachev
>
> Texas Tech University, 2026

The current draft is in [`paper_overleaf/`](paper_overleaf/) (`Memory_Roughness_Persistence.tex`).

## Headline Findings

Panel of 115 S&P 500 constituents, 6,136 trading days (29 Nov 2001 – 21 Apr 2026), Bloomberg.

| | Value |
|---|---|
| Long memory of Parkinson RV (GPH) | $\hat d = 0.226$ (98% sig.) |
| Long memory of Parkinson RV (Local Whittle) | $\hat d = 0.440$ (100% sig.) |
| Roughness of $\log\mathrm{RV}^{PK}$ (Hurst) | $H = 0.063$, all 115 stocks $H<\tfrac12$ |
| Cross-sectional mean $\bar d_t$ vs calm 2013-14 baseline | +68% in GFC, +86% in COVID |
| Correlation $\rho(\bar d_t,\,\mathrm{VIX})$ | $+0.50$ |
| Model C vs HAR (pooled MSE on $\log\mathrm{RV}^{PK}$, $h\in\{1,5,22\}$) | **+4.6%, +8.2%, +5.5%** |
| HLN-corrected DM-$t$ (panel-aware) | +3.15, +3.87, +2.33 |
| Vol-managed portfolio Sharpe in COVID (C / unmanaged) | **1.37 / 0.65** |
| Vol-managed CER in COVID, $\gamma=5$ (C / unmanaged) | **+12.3% / −3.0%** |

The layered ablation $A \to A_1 \to A_2 \to \dots \to A_5 \to C \to D$ shows
that VIX/MOVE controls (HAR-X $= A_1$) capture most of the pooled gain,
while the persistence aggregates contribute incrementally at the monthly
horizon (+1.8 pp beyond HAR-X) and in stress regimes (+1.9 pp at $h=5$
high-VIX, +1.7 pp at $h=22$ in COVID). Tree-based ML estimators with the
same predictors fail to match the linear Model $C$.

## Layered Forecasting Ladder

Each model is identified by its predictor set; estimator-only differences
are factored out into Model $D$.

| Model | Predictors | $n$ |
|---|---|---|
| $A$ | HAR core: $\log\mathrm{RV}^{(d,w,m)}$, $r_{t-1}$, $\lvert r_{t-1}\rvert$ | 5 |
| $A_1$ (HAR-X) | $A$ + VIX, MOVE | 7 |
| $A_2$ | $A$ + own-stock $\hat d$ block ($\hat d, \Delta\hat d, \mathrm{Vol}(\hat d), \mathrm{Trend}(\hat d), H, \Delta H$) | 11 |
| $A_3$ | $A$ + cross-sectional mean and dispersion of $\hat d$ | 7 |
| $A_4$ | $A$ + sector-mean $\hat d$ | 6 |
| $A_5$ | $A$ + $\hat d$, VIX, MOVE, $\hat d\times$VIX, $\hat d\times$MOVE | 10 |
| $C$ | Full union of $A_1$–$A_5$ | 18 |
| $D$ | Same predictors as $C$, estimated by Lasso / Ridge / Elastic Net / Random Forest / Gradient Boosting | 18 |

## Repository Structure

```
LRD-ML/
├── README.md
├── PROJECT_LOG.md          # session-by-session research log
├── WORK_PLAN.md            # original modular plan
├── PHASE4_RESULTS.md       # Phase 4 forecasting summary
├── SESSION_REPORT.md       # mid-project handoff
├── requirements.txt
├── .gitignore              # excludes bloomberg_pull/, *.csv, *.docx
│
├── modules/
│   ├── io_v2.py                            # single source of truth for the panel
│   ├── module1_data_description.py         # Tables 1–2, Figure 1
│   ├── module1b_preliminary_diagnostics.py # FIGARCH table, diagnostic figure
│   ├── module2_lrd_estimation.py           # GPH, Local Whittle, Hurst, Table 3, Figure 2
│   ├── module3_feature_engineering.py      # Persistence feature vector, Table 4
│   ├── forecast_io.py                      # feature/target assembly for Modules 4–6
│   ├── module4_benchmarks.py               # Linear ladder forecasts (A, A1..A5, C)
│   ├── module5_ml_models.py                # ML ladder (D_lasso, D_ridge, D_en, D_rf, D_gbm)
│   ├── module6_forecast_eval.py            # HLN-corrected DM, Tables 5–8
│   ├── module9_robustness.py               # Robustness summary (Table 9)
│   ├── module10_plots.py                   # Figures 4–8 (cumulative loss, ablation, etc.)
│   └── module11_economic.py                # Vol-managed portfolios (Table 10, Figure 9)
│
├── preprocess_bloomberg.py                 # Clean Bloomberg OHLCV
├── preprocess_supporting.py                # Clean market-level + sector ETF + Parkinson RV
│
├── bloomberg_pull/                         # raw + processed Bloomberg data (GITIGNORED)
│
├── paper_overleaf/                         # current manuscript bundle
│   ├── Memory_Roughness_Persistence.tex
│   ├── refs.bib
│   ├── figures/
│   └── tables/
│
└── results/
    ├── tables/                             # tables 1–10 + FIGARCH
    ├── figures/                            # figures 1–9
    └── intermediate/
        ├── forecasts/                      # per-model, per-horizon yhat/y panels
        └── features/                       # Z_t feature panels
```

## Reproducing the Analysis

The Bloomberg-sourced inputs under `bloomberg_pull/` are gitignored. With
those in place, the full pipeline is:

```bash
# Data layer
python preprocess_bloomberg.py
python preprocess_supporting.py
python -m modules.io_v2                       # builds bloomberg_pull/processed/clean_panel/

# Estimation + features
python -m modules.module1_data_description
python -m modules.module1b_preliminary_diagnostics
python -m modules.module2_lrd_estimation
python -m modules.module3_feature_engineering

# Forecasting
python -m modules.module4_benchmarks          # linear ladder A, A1..A5, C
python -m modules.module5_ml_models           # D_lasso, D_ridge, D_en, D_rf, D_gbm

# Evaluation
python -m modules.module6_forecast_eval       # Tables 5, 6, 7, 8 + HLN-DM
python -m modules.module9_robustness          # Table 9 (window/estimator/liquidity/target)
python -m modules.module10_plots              # Figures 4–8
python -m modules.module11_economic           # Table 10, Figure 9 (vol-managed portfolios)
```

Output tables drop into `results/tables/`, figures into `results/figures/`,
and intermediate panels into `results/intermediate/`. The
`paper_overleaf/` bundle is hand-synced from those outputs.

## Methodology Highlights

**Persistence estimators.** Geweke-Porter-Hudak log-periodogram regression
and Robinson local Whittle, both at bandwidth $m = \lfloor T^{0.65}\rfloor$
on a 750-day rolling window with weekly stride. Hurst exponent estimated
on $\log\mathrm{RV}^{PK}$ via the scaling of the $q=2$ moment of
increments over lags $\{1,2,3,5,8,13,21\}$.

**Forecast target.** $y_{i,t,h} = \log\!\bigl(\tfrac{1}{h}\sum_{j=1}^{h}
\mathrm{RV}^{PK}_{i,t+j}\bigr)$, the log mean future Parkinson range-based
variance proxy over horizon $h\in\{1,5,22\}$.

**Inference.** Diebold-Mariano with the Harvey-Leybourne-Newbold (1997)
finite-sample correction, computed on the cross-sectional mean loss
differential per date with a Newey-West HAC variance (bandwidth
$\lceil h/5\rceil - 1$ to handle multi-step overlap on the weekly stride)
and Student-$t(T-1)$ reference. The panel-aware HLN stat is roughly $6\times$
smaller than the iid pooled-cell stat, which inflates $t$-values by
ignoring within-date cross-sectional dependence.

**Economic significance.** Moreira-Muir (2017) volatility-managed
portfolios with weights $w_{m,i,t} = c_{m,i}/\hat\sigma^2_{m,i,t}$,
per-stock variance normalisation, and weekly rebalancing aligned to the
forecast sample stride.

**Robustness.** Table 9 confirms the headline against alternative
estimator (LW), alternative windows (500, 1000), alternative target
(squared returns), and liquidity-bucket subsamples; all yield +4–9% MSE
gains with HLN-DM-$t > 3$.

## Citation

```bibtex
@article{deep2026memory,
  title   = {Memory, Roughness, and Information Persistence in Financial Markets:
             A Structural and Machine Learning Approach to Volatility Forecasting},
  author  = {Deep, Akash and Appiah, Nicholas and Rachev, Svetlozar T.},
  journal = {Working Paper},
  year    = {2026},
  institution = {Texas Tech University}
}
```

## Data Notes

- Stock-level OHLCV + market-level series (VIX, MOVE, rates, CDX, sector
  ETFs) sourced from **Bloomberg Terminal**. Raw and processed panels live
  under `bloomberg_pull/` and are gitignored.
- Daily history starts 29 November 2001 (Bloomberg's earliest for the
  full panel).
- A 5-minute intraday slice (October 2025 – April 2026) is available for
  20 liquid stocks and the index, used as a high-frequency robustness
  reference.

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Advisor and co-author: Prof. Svetlozar T. Rachev, Department of
Mathematics and Statistics, Texas Tech University.

## Contact

- Akash Deep — akash.deep@ttu.edu
- Nicholas Appiah — Texas Tech University
- Svetlozar T. Rachev — Texas Tech University
