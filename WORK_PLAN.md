# LRD-ML Research: Modular Work Plan

**Goal:** Produce a publication-ready paper for *Journal of Financial Econometrics* Special Issue
**Deadline:** March 1, 2026
**Target:** "Long-Range Dependence as Informative Features for Machine Learning in Financial Forecasting"

---

## Paper Structure & Required Outputs

| Section | Key Content | Required Outputs |
|---------|-------------|------------------|
| 1. Introduction | Motivation, contribution | - |
| 2. Literature Review | LRD in finance, ML forecasting | - |
| 3. Methodology | LRD estimation, feature construction, models | Equations, algorithm boxes |
| 4. Data | Sample description, summary stats | **Table 1-2, Figure 1** |
| 5. Empirical Results | Main findings | **Tables 3-6, Figures 2-5** |
| 6. Robustness | Alternative specs, subsamples | **Tables 7-8** |
| 7. Conclusion | Summary, implications | - |

---

## Module Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WORK PLAN MODULES                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MODULE 1: Data Description          ──► Table 1-2, Figure 1       │
│     └── Summary stats, coverage, stylized facts                    │
│                                                                     │
│  MODULE 2: LRD Estimation            ──► Table 3, Figure 2         │
│     └── GPH, Local Whittle, rolling d-hat                          │
│                                                                     │
│  MODULE 3: Feature Engineering       ──► Table 4                   │
│     └── Memory dynamics, cross-sectional features                  │
│                                                                     │
│  MODULE 4: Benchmark Models          ──► Table 5 (partial)         │
│     └── HAR-RV, FIGARCH, Pure ML                                   │
│                                                                     │
│  MODULE 5: Hybrid LRD-ML Models      ──► Table 5-6, Figure 3-4     │
│     └── XGBoost + LRD features, LSTM                               │
│                                                                     │
│  MODULE 6: Statistical Evaluation    ──► Table 5 (complete)        │
│     └── DM tests, Model Confidence Set                             │
│                                                                     │
│  MODULE 7: Interpretation            ──► Figure 5, Table 6         │
│     └── SHAP values, feature importance                            │
│                                                                     │
│  MODULE 8: Robustness                ──► Tables 7-8                 │
│     └── Subsamples, horizons, alternative specs                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Module Specifications

---

### MODULE 1: Data Description
**Status:** 🟡 In Progress (preprocessing done)
**Priority:** HIGH (do first - sets up everything else)

#### Tasks
- [ ] 1.1 Summary statistics for returns (mean, std, skew, kurtosis)
- [ ] 1.2 Summary statistics for volatility proxy
- [ ] 1.3 Sector breakdown of sample
- [ ] 1.4 Time series plots (sample stocks, VIX)
- [ ] 1.5 Return distribution analysis (fat tails, non-normality)
- [ ] 1.6 Autocorrelation analysis (ACF of returns vs squared returns)

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Table 1** | Summary statistics (125 stocks, panel) | `results/tables/table1_summary_stats.tex` |
| **Table 2** | Sector breakdown & macro variables | `results/tables/table2_data_description.tex` |
| **Figure 1** | (a) Sample return series, (b) Squared returns ACF, (c) VIX | `results/figures/fig1_data_overview.pdf` |

#### Code
```
module1_data_description.py
├── compute_summary_stats()      → Table 1
├── compute_sector_breakdown()   → Table 2
├── plot_sample_series()         → Figure 1a
├── plot_acf_comparison()        → Figure 1b
└── export_latex_tables()
```

---

### MODULE 2: LRD Estimation
**Status:** 🔴 Not Started
**Priority:** HIGH (core methodology)

#### Tasks
- [ ] 2.1 Implement GPH estimator with standard errors
- [ ] 2.2 Implement Local Whittle estimator
- [ ] 2.3 Rolling window estimation (250-day, 500-day windows)
- [ ] 2.4 Cross-sectional distribution of d-hat
- [ ] 2.5 Time series of d-hat for selected stocks
- [ ] 2.6 Validate against known benchmarks (simulate ARFIMA)

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Table 3** | LRD estimates by sector (mean, std of d-hat) | `results/tables/table3_lrd_estimates.tex` |
| **Figure 2** | (a) Distribution of d-hat, (b) Rolling d-hat for SPY/VIX, (c) Cross-sectional dispersion over time | `results/figures/fig2_lrd_estimates.pdf` |

#### Code
```
module2_lrd_estimation.py
├── gph_estimator(x, m)          → d_hat, se
├── local_whittle(x, m)          → d_hat, se
├── rolling_lrd_estimate()       → time series of d_hat
├── cross_sectional_lrd()        → d_hat for all stocks
├── plot_lrd_distribution()      → Figure 2a
├── plot_rolling_d()             → Figure 2b
└── export_results()
```

#### Key Equations for Paper
```latex
% GPH Estimator
\hat{d}_{GPH} = -\frac{1}{2} \frac{\sum_{j=1}^m (x_j - \bar{x})(\ln I(\lambda_j) - \overline{\ln I})}{\sum_{j=1}^m (x_j - \bar{x})^2}

% Local Whittle
\hat{d}_{LW} = \arg\min_d Q(d) = \frac{1}{m}\sum_{j=1}^m \left[\ln(\lambda_j^{-2d} \hat{G}) + \frac{I(\lambda_j)}{\lambda_j^{-2d}\hat{G}}\right]
```

---

### MODULE 3: Feature Engineering
**Status:** 🔴 Not Started
**Priority:** HIGH (novel contribution)

#### Tasks
- [ ] 3.1 Construct LRD feature vector (d-hat, SE, residuals)
- [ ] 3.2 Memory dynamics features (Δd, Vol(d), Trend(d))
- [ ] 3.3 Cross-sectional memory features (mean, std, skew of d)
- [ ] 3.4 HAR components (RV daily, weekly, monthly)
- [ ] 3.5 Combine all features into unified dataset
- [ ] 3.6 Feature correlation analysis

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Table 4** | Feature definitions and summary statistics | `results/tables/table4_features.tex` |

#### Code
```
module3_feature_engineering.py
├── compute_lrd_features()       → d_hat, SE, z_t
├── compute_memory_dynamics()    → Δd, Vol(d), Trend(d)
├── compute_cross_sectional()    → mean_d, std_d, skew_d
├── compute_har_components()     → RV_d, RV_w, RV_m
├── build_feature_matrix()       → X (all features combined)
└── export_feature_stats()       → Table 4
```

#### Feature Vector (for paper)
```latex
\mathbf{F}_t^{LRD} = \begin{pmatrix}
    \hat{d}_t & \text{SE}(\hat{d}_t) & \Delta\hat{d}_t &
    \text{Vol}(\hat{d})_t & z_t^2 & \sigma_t^2
\end{pmatrix}^\top
```

---

### MODULE 4: Benchmark Models
**Status:** 🔴 Not Started
**Priority:** HIGH

#### Tasks
- [ ] 4.1 HAR-RV model implementation
- [ ] 4.2 FIGARCH(1,d,1) implementation (use arch package)
- [ ] 4.3 Pure ML baseline (XGBoost on raw features)
- [ ] 4.4 Rolling window forecasts for all benchmarks
- [ ] 4.5 Store predictions for evaluation

#### Paper Outputs
Contributes to **Table 5** (model comparison)

#### Code
```
module4_benchmarks.py
├── fit_har_rv()                 → HAR forecasts
├── fit_figarch()                → FIGARCH forecasts
├── fit_pure_ml()                → Pure XGBoost forecasts
├── rolling_forecast()           → out-of-sample predictions
└── save_predictions()
```

---

### MODULE 5: Hybrid LRD-ML Models
**Status:** 🔴 Not Started
**Priority:** HIGH (main contribution)

#### Tasks
- [ ] 5.1 XGBoost with LRD features
- [ ] 5.2 LSTM with LRD feature inputs
- [ ] 5.3 Residual learning (ML on FIGARCH residuals)
- [ ] 5.4 Hyperparameter tuning (time-series CV)
- [ ] 5.5 Rolling window forecasts

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Figure 3** | Actual vs predicted volatility (sample period) | `results/figures/fig3_predictions.pdf` |
| **Figure 4** | Crisis period performance (2008, 2020) | `results/figures/fig4_crisis.pdf` |

#### Code
```
module5_hybrid_models.py
├── fit_lrd_xgboost()            → XGBoost + LRD features
├── fit_lrd_lstm()               → LSTM + LRD features
├── fit_residual_learner()       → ML on FIGARCH residuals
├── hyperparameter_search()      → Bayesian optimization
├── rolling_forecast()           → out-of-sample predictions
└── plot_predictions()           → Figure 3-4
```

---

### MODULE 6: Statistical Evaluation
**Status:** 🔴 Not Started
**Priority:** HIGH

#### Tasks
- [ ] 6.1 Compute loss functions (MSE, QLIKE, MAE)
- [ ] 6.2 Diebold-Mariano tests (pairwise)
- [ ] 6.3 Model Confidence Set
- [ ] 6.4 Create comprehensive comparison table

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Table 5** | Model comparison (losses + DM tests) | `results/tables/table5_model_comparison.tex` |

#### Code
```
module6_evaluation.py
├── compute_losses()             → MSE, QLIKE, MAE
├── diebold_mariano_test()       → DM statistics, p-values
├── model_confidence_set()       → MCS p-values
├── create_comparison_table()    → Table 5
└── export_latex()
```

#### Table 5 Structure
```
| Model          | MSE    | QLIKE  | MAE    | DM vs HAR | MCS p-val |
|----------------|--------|--------|--------|-----------|-----------|
| HAR-RV         | x.xxx  | x.xxx  | x.xxx  | -         | x.xxx     |
| FIGARCH        | x.xxx  | x.xxx  | x.xxx  | x.xx**    | x.xxx     |
| Pure ML        | x.xxx  | x.xxx  | x.xxx  | x.xx**    | x.xxx     |
| LRD-XGBoost    | x.xxx  | x.xxx  | x.xxx  | x.xx***   | x.xxx     |
| LRD-LSTM       | x.xxx  | x.xxx  | x.xxx  | x.xx***   | x.xxx     |
| Residual ML    | x.xxx  | x.xxx  | x.xxx  | x.xx***   | x.xxx     |
```

---

### MODULE 7: Interpretation
**Status:** 🔴 Not Started
**Priority:** MEDIUM

#### Tasks
- [ ] 7.1 SHAP values for XGBoost
- [ ] 7.2 Feature importance ranking
- [ ] 7.3 Partial dependence plots
- [ ] 7.4 Interaction analysis (d-hat × VIX)

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Figure 5** | (a) SHAP summary, (b) Feature importance, (c) PDP for d-hat | `results/figures/fig5_interpretation.pdf` |
| **Table 6** | Top 10 features by importance | `results/tables/table6_feature_importance.tex` |

#### Code
```
module7_interpretation.py
├── compute_shap_values()        → SHAP analysis
├── plot_shap_summary()          → Figure 5a
├── plot_feature_importance()    → Figure 5b
├── plot_partial_dependence()    → Figure 5c
└── export_importance_table()    → Table 6
```

---

### MODULE 8: Robustness Checks
**Status:** 🔴 Not Started
**Priority:** MEDIUM

#### Tasks
- [ ] 8.1 Subsample analysis (pre-2008, 2008-2015, 2016-2024)
- [ ] 8.2 Different forecast horizons (h=1, 5, 22 days)
- [ ] 8.3 Alternative d-hat estimators (GPH vs LW vs ELW)
- [ ] 8.4 Different window sizes (250 vs 500 vs 1000 days)
- [ ] 8.5 Sector-level analysis

#### Paper Outputs
| Output | Description | File |
|--------|-------------|------|
| **Table 7** | Subsample results | `results/tables/table7_subsamples.tex` |
| **Table 8** | Forecast horizon comparison | `results/tables/table8_horizons.tex` |

#### Code
```
module8_robustness.py
├── subsample_analysis()         → Table 7
├── horizon_comparison()         → Table 8
├── estimator_comparison()       → sensitivity analysis
└── window_size_analysis()
```

---

## Execution Order

### Phase 1: Foundation (Week 1-2)
```
MODULE 1 → MODULE 2 → MODULE 3
   ↓           ↓           ↓
Table 1-2   Table 3    Table 4
Figure 1    Figure 2
```

### Phase 2: Models (Week 3-4)
```
MODULE 4 → MODULE 5 → MODULE 6
   ↓           ↓           ↓
Benchmarks  Hybrids    Table 5
           Figure 3-4
```

### Phase 3: Analysis (Week 5-6)
```
MODULE 7 → MODULE 8
   ↓           ↓
Figure 5   Table 7-8
Table 6
```

### Phase 4: Paper Writing (Week 7+)
```
Compile all outputs → Draft paper → Revisions
```

---

## Current Priority: MODULE 1

**Start with:** `module1_data_description.py`

This produces the foundation for everything else and gives us:
- Table 1: Summary statistics (needed for Section 4)
- Figure 1: Data visualization (needed for Section 4)
- Verification that our processed data is correct

---

## File Organization

```
LRD_Nicholas/
├── modules/
│   ├── module1_data_description.py
│   ├── module2_lrd_estimation.py
│   ├── module3_feature_engineering.py
│   ├── module4_benchmarks.py
│   ├── module5_hybrid_models.py
│   ├── module6_evaluation.py
│   ├── module7_interpretation.py
│   └── module8_robustness.py
│
├── results/
│   ├── tables/
│   │   ├── table1_summary_stats.tex
│   │   ├── table2_data_description.tex
│   │   └── ...
│   ├── figures/
│   │   ├── fig1_data_overview.pdf
│   │   ├── fig2_lrd_estimates.pdf
│   │   └── ...
│   └── intermediate/
│       ├── lrd_estimates.pkl
│       ├── features.pkl
│       └── predictions.pkl
│
└── paper/
    ├── main.tex
    ├── references.bib
    └── appendix.tex
```

---

## Progress Tracker

| Module | Status | Tables | Figures | Last Updated |
|--------|--------|--------|---------|--------------|
| 1. Data Description | 🟢 | 2/2 | 1/1 | 2025-01-21 |
| 2. LRD Estimation | 🟢 | 1/1 | 1/1 | 2025-01-21 |
| 3. Feature Engineering | 🟢 | 1/1 | 0/0 | 2025-01-21 |
| 4. Benchmarks | 🟢 | 0/0 | 0/0 | 2025-01-21 |
| 5. Hybrid Models | 🟢 | 0/0 | 0/2 | 2025-01-21 |
| 6. Evaluation | 🟢 | 1/1 | 0/0 | 2025-01-21 |
| 7. Interpretation | 🟢 | 1/1 | 1/1 | 2025-01-21 |
| 8. Robustness | 🟡 | 0/2 | 0/0 | - |

**Legend:** 🟢 Complete | 🟡 In Progress | 🔴 Not Started

---

*Last updated: 2025-01-21*
