# LRD-ML: Long-Range Dependence as Informative Features for Machine Learning in Financial Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and results for the research paper:

> **Long-Range Dependence as Informative Features for Machine Learning in Financial Forecasting**
>
> Akash Deep, Nicholas Appiah
>
> Texas Tech University, 2025

We investigate whether estimated parameters from Long-Range Dependence (LRD) models contain predictive information for volatility forecasting beyond their traditional role in stationarity transformation.

## Key Findings

| Finding | Result |
|---------|--------|
| Volatility memory parameter | d̂ ≈ 0.21 (GPH), 100% significant |
| Forecast improvement | 9/10 stocks improved, avg 3.4% MSE reduction |
| Cross-sectional memory importance | 2nd most important feature (13.8%) |
| LRD features total importance | 37.1% of predictive power |
| Best performance | High-volatility regimes (COVID +4.9%, Crisis +2.7%) |

## Repository Structure

```
LRD-ML/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── modules/                  # Core analysis modules
│   ├── module1_data_description.py    # Summary statistics, Figure 1
│   ├── module2_lrd_estimation.py      # GPH/Local Whittle estimators
│   ├── module3_feature_engineering.py # LRD feature construction
│   ├── module4_benchmarks.py          # HAR-RV benchmark
│   ├── module7_interpretation.py      # Feature importance
│   └── module8_robustness.py          # Subsample & horizon analysis
│
├── download_free_data.py     # Download data from Yahoo Finance & FRED
├── preprocess_data.py        # Data cleaning and alignment
│
├── paper/                    # LaTeX manuscript
│   └── LRD_ML_Paper.tex     # Complete paper
│
├── results/
│   ├── tables/              # LaTeX tables (Tables 1-8)
│   └── figures/             # PDF/PNG figures (Figures 1, 2, 5)
│
├── PROJECT_LOG.md           # Detailed research log
└── WORK_PLAN.md             # Modular work plan
```

## Installation

```bash
# Clone the repository
git clone https://github.com/akashdeepo/LRD-ML.git
cd LRD-ML

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Free Data

```bash
python download_free_data.py
```

This downloads:
- 137 S&P 500 stocks (daily prices, 2000-2025)
- VIX index
- Market indices (SPX, DJIA, NASDAQ)
- Macro variables from FRED

### 2. Preprocess Data

```bash
python preprocess_data.py
```

Outputs cleaned data to `processed/` folder.

### 3. Run Analysis Modules

```bash
# Run all modules in sequence
python modules/module1_data_description.py    # Table 1-2, Figure 1
python modules/module2_lrd_estimation.py      # Table 3, Figure 2
python modules/module3_feature_engineering.py # Table 4
python modules/module4_benchmarks.py          # HAR-RV benchmarks
python modules/module7_interpretation.py      # Table 6, Figure 5
python modules/module8_robustness.py          # Tables 7-8
```

## Methodology

### LRD Estimators

We implement two semiparametric estimators:

1. **GPH (Geweke-Porter-Hudak)**: Log-periodogram regression
2. **Local Whittle**: Gaussian semiparametric MLE

### Feature Engineering

Novel LRD-based features:
- Rolling memory parameter (d̂)
- Memory dynamics (Δd̂, Vol(d̂), Trend(d̂))
- Cross-sectional features (mean, std, skew of d̂ across stocks)

### Models

- **HAR-RV**: Benchmark (Corsi, 2009)
- **HAR-LRD**: HAR-RV + LRD features (linear)
- **XGBoost-LRD**: Nonlinear model with LRD features

## Results

### Table 5: Model Comparison

| Stock | MSE (HAR) | MSE (LRD) | Improvement | DM Stat |
|-------|-----------|-----------|-------------|---------|
| AAPL  | 4.89      | 4.63      | 5.2%        | 1.69*   |
| MSFT  | 2.18      | 2.02      | 7.5%        | 0.87    |
| WMT   | 5.79      | 5.31      | 8.3%        | 1.09    |
| ...   | ...       | ...       | ...         | ...     |
| **Avg** | **6.13** | **5.94** | **3.4%**   | -       |

*MSE × 10⁷. 9/10 stocks show improvement.*

### Table 7: Subsample Analysis

| Period | Improvement |
|--------|-------------|
| Financial Crisis (2008-2009) | +2.7% |
| Recovery (2010-2015) | -0.9% |
| Bull Market (2016-2019) | +1.0% |
| COVID & After (2020-2024) | +4.9% |

*LRD features most valuable during high-volatility regimes.*

## Citation

```bibtex
@article{deep2025lrd,
  title={Long-Range Dependence as Informative Features for Machine Learning in Financial Forecasting},
  author={Deep, Akash and Appiah, Nicholas},
  journal={Working Paper},
  year={2025},
  institution={Texas Tech University}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Advisor: Dr. Svetlozar Rachev, Texas Tech University
- Data: Yahoo Finance, FRED

## Contact

- Akash Deep: akash.deep@ttu.edu
- Nicholas Appiah: Texas Tech University
