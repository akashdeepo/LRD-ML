# Experiment ledger

One entry per scored run, newest first. Written before the run (header, setup,
hypothesis) and completed when the numbers arrive; past entries are never
edited, they are superseded by new ones. Findings are numbered in
[FINDINGS.md](FINDINGS.md) and cited here by number.

**Floor.** HAR (Model A, 5 predictors) and HAR-X (Model A1 = HAR + VIX, MOVE).
Every forecasting run is scored against both. Metric: pooled out-of-sample MSE
on log mean future Parkinson variance, horizons h in {1, 5, 22}; inference by
the panel-aware HLN-corrected Diebold-Mariano test (module 6) and, for nested
comparisons, Clark-West (2007). "Local" = pooled full evaluation window
(2013-06 to 2026-04); "external" = the three regime sub-windows (low-VIX Q1,
high-VIX Q4, COVID Mar-Dec 2020), which are small and reported without
promotion power.

**Pre-registered protocol for the 2026-09-14 runs (written before any result).**
1. Fix the training-window leak (finding #1): at origin t, a row j may enter
   the training set only if its target window ends on or before the origin
   trading day, pos_j + h <= pos_t. Applied to modules 4, 5, 9.
2. Re-run every model at h = 22 (the only affected horizon) and Model C at
   h = 5 as a regression check (must reproduce the saved file exactly).
3. Three new linear specifications, fixed in advance, all nesting HAR-X:
   - `A1cs`  = HAR-X + cross-sectional mean and dispersion of d-hat (9 predictors)
   - `A1sec` = HAR-X + sector-mean d-hat (8)
   - `A1mod` = HAR-X + d-bar_t + (log RV_d, log RV_w, log RV_m) x d-bar_t (11):
     the "duration" hypothesis, persistence modulating the HAR weights.
   No other specification will be added after seeing results.
4. Tests against HAR-X at each horizon: HLN-DM (two-sided) and Clark-West
   (one-sided, nested), both on the cross-sectional mean loss differential per
   date with Newey-West HAC bandwidth ceil(h/5)-1. Nine CW tests in total;
   Holm-adjusted p-values reported alongside raw ones.
5. Decision rule: a specification "adds information beyond HAR-X" at a
   horizon only if CW p < 0.05 after Holm adjustment AND the pooled MSE gain
   over HAR-X is positive. DM significance is reported as the stricter
   criterion. Regime results are descriptive.
6. Volatility-managed portfolios (h = 5, unaffected by the leak): Sharpe-ratio
   differences C-managed vs HAR-X-managed and vs unmanaged, full sample and
   COVID, with the Ledoit-Wolf (2008) HAC test and a circular block bootstrap.

**Pre-registered protocol, second block (written 2026-09-14 after runs 01-04,
before any of the runs below).** Same floor, metric, tests and decision rule
as above (Holm-adjusted Clark-West p < 0.05 AND positive MSE gain against the
run's own HAR-X-type baseline; DM reported as the stricter criterion).
Motivation: Clark-West rejects while MSE does not improve (finding #7), so
the information is there and per-stock estimation noise is the suspect.

- run-05 (pooling). Estimate each specification ONCE on the stacked panel of
  all stocks with stock fixed effects (within transformation using training
  means only), expanding window, refit at every origin, same embargo.
  Specifications: P-A, P-A1, P-C, P-A1cs at h = 1, 5, 22. Comparisons:
  P-C vs P-A1 and P-A1cs vs P-A1 (does pooling unlock the persistence
  information?), and P-A1 vs per-stock A1 (does pooling help HAR-X itself?).
  Six CW tests Holm-adjusted.
- run-06 (duration). Target D_t = y_{t,22} - y_{t,5}, the log ratio of monthly
  to weekly future mean Parkinson variance (how slowly the next month's
  variance decays relative to next week's). Per-stock expanding OLS with the
  h = 22 embargo. Specifications A, A1, A1cs, C on this target. Comparisons:
  A1cs vs A1, C vs A1. Two CW tests Holm-adjusted. If persistence measures
  duration, it should help here even though it does not help the level.
- run-07 (market level). One series: the cross-sectional mean of the stock
  targets, y^M_{t,h} = mean_i y_{i,t,h}. Market HAR components from the
  cross-sectional mean daily log RV (same t-1 convention), VIX, MOVE, and the
  persistence state (d-bar_t, sigma_d, d-bar x VIX, d-bar x MOVE).
  Specifications M-A, M-A1, M-C at h = 1, 5, 22; expanding OLS with embargo.
  Comparison M-C vs M-A1, three CW tests Holm-adjusted. Low power by
  construction (one series, ~645 origins); reported as such.
- run-08 (point-in-time winsorisation). Returns are winsorised at expanding
  0.1%/99.9% quantiles (minimum 500 observations) instead of full-sample
  quantiles; the portfolio exercise uses raw returns. Expect negligible
  changes; every table is regenerated from this panel so the whole paper is
  point-in-time. Reported as a reproduction check (max change in any
  headline number).
- run-09 (real-time portfolio normalisation). The Moreira-Muir constant
  c_{m,i} is computed from data through t-1 (expanding, 52-week warm-up)
  instead of the full evaluation sample. Headline Table 10 and the Sharpe
  tests use the real-time version; the full-sample version is kept as a
  robustness row.
- Documentation only: S&P 500 membership (constituents at the April 2026
  pull; no point-in-time reconstruction), HAR components at t-1 (notation).

Nothing else will be added to this block after results are seen.

---

## 2026-09-14  run-09  Real-time portfolio normalisation
- Setup: `module11_economic.REALTIME_C = True`, c_{i,t} from the expanding
  window of weeks before t with a 52-week warm-up (evaluation window 593
  weeks instead of 645); full-sample-c version kept as a table row; raw
  returns for portfolio P&L (run-08). Sharpe tests re-run (module 12 part B).
- Result: annualised Sharpe, full sample: unmanaged 0.61, HAR-X-managed
  0.63, C-managed 0.66 (full-sample c: 0.59). COVID: 0.50 / 1.13 / 1.41
  (full-sample c 1.27). Ledoit-Wolf tests: full-sample C vs HAR-X +0.025,
  p = 0.31 (bootstrap 0.29); COVID +0.30, p = 0.15 (bootstrap 0.37);
  high-VIX +0.07, p = 0.29. Nothing significant.
- Verdict: KEEP as the headline construction (implementable); conclusions
  of run-04 unchanged (finding #8 stands).
- Cost: seconds.
- Lesson: the full-sample constant flattered nothing systematically; the
  real-time version is slightly better for C and slightly worse overall.

## 2026-09-14  run-08  Point-in-time winsorisation (reproduction check)
- Setup: `io_v2._winsorize_pit` (expanding 0.1%/99.9% quantiles, 500-obs
  warm-up; 1,470 of 698,837 return cells clipped), raw returns kept for the
  portfolio exercise; entire pipeline re-run from module 4 (ML in progress
  at time of writing; linear rows below).
- Result: headline numbers move by at most 0.01 pp: C vs HAR +4.55/+8.25/-0.62%
  (was +4.55/+8.24/-0.62), HLN-t 3.15/3.88/-0.21; HAR-X +5.10/+7.73/+2.82%.
  Table 9 headline +8.25% (was +8.24%); GARCH row -95.9% (was -93.5%,
  GARCH refit on the new returns). HAR-X tests (Table 11) identical to two
  decimals.
- Verdict: KEEP (the paper is now point-in-time end to end); no finding
  changes.
- Cost: ~20 min linear; ML ~3 h.
- Lesson: full-sample winsorisation was a disclosure problem, not a
  results problem.

## 2026-09-14  run-07  Market-level target: does the persistence state help forecast market variance?
- Setup: one series, y^M = cross-sectional mean of the stock targets;
  M-A (market HAR), M-A1 (+VIX, MOVE), M-C (+d-bar, sigma_d, d-bar x VIX,
  d-bar x MOVE); expanding OLS with embargo; `module13_candidates.run07`.
- Result (M-C vs M-A1): h=1 -1.22% (DM -0.49, CW +1.72, Holm p 0.13);
  h=5 -1.90% (DM -0.93, CW +1.09, p 0.28); h=22 -6.66% (DM -3.13, CW -2.22).
  For reference M-A1 vs M-A: +11.5% / +10.9% / +0.7%.
- Verdict: REVERT. At the market level the persistence state adds nothing
  to VIX and MOVE and hurts at the monthly horizon (finding #11).
- Cost: seconds.
- Lesson: the co-movement of d-bar with the VIX (rho 0.50) is the whole
  story at the index level.

## 2026-09-14  run-06  Duration target: does persistence predict how slowly variance decays?
- Setup: target D_t = y_{t,22} - y_{t,5}; per-stock expanding OLS with the
  h=22 embargo; A, A1, A1cs, C on this target; `module13_candidates.run06`.
- Result: A1cs vs A1 -0.13% (DM -0.44, CW +1.07, Holm p 0.14);
  C vs A1 -1.80% (DM -3.12, CW +2.34, p 0.02, but MSE worse so fails the
  rule). A1 vs A +0.14%.
- Verdict: REVERT. The persistence state does not forecast the term-structure
  slope of future variance beyond HAR-X; the "duration" interpretation is
  not supported as a forecastable quantity (finding #10).
- Cost: seconds.
- Lesson: this is the cleanest test of the paper's economic story and it
  is negative; the paper must say so.

## 2026-09-14  run-05  Pooled panel with stock fixed effects
- Setup: one regression per origin on the stacked panel (within
  transformation on training rows), embargo, P-A, P-A1, P-C, P-A1cs at
  h = 1, 5, 22; `module13_candidates.run05`; tests P-C vs P-A1 and
  P-A1cs vs P-A1 (Holm over 6), plus P-A1 vs per-stock A1.
- Result: P-C vs P-A1: h=1 +0.20% (DM +0.36, CW +2.55, Holm p 0.033,
  passes the rule); h=5 +0.19% (DM +0.25, CW +2.27, Holm p 0.058);
  h=22 -1.61% (DM -1.31). P-A1cs vs P-A1: -0.05% / -0.36% / -1.57%.
  Pooling itself: P-A1 vs per-stock A1 +0.06% / -0.72% (DM -2.18) / -0.01%.
- Verdict: KEEP as a result, but it does not change the paper's answer:
  pooling recovers a statistically detectable and economically negligible
  gain (+0.2% MSE) at the daily horizon only (finding #9). Not promoted to
  a headline specification.
- Cost: ~6 min (12 model-horizon fits on a 74k-row panel, 646 refits each).
- Lesson: estimation noise was part of the story, but the recoverable
  information is tiny; the persistence state is nearly redundant with VIX
  and MOVE for stock-level HAR forecasting.

## 2026-09-14  run-04  Are the vol-managed Sharpe differences distinguishable from zero?
- Setup: protocol step 6; `module12_incremental_tests.part_b` on the h = 5
  forecasts (unaffected by the leak); Ledoit-Wolf HAC delta method plus a
  studentised circular block bootstrap (block 5, B = 2000, seed 0).
- Result (annualised Sharpe difference, HAC p, bootstrap p): full sample
  C vs HAR-X +0.01 (0.62, 0.59); C vs unmanaged -0.01 (0.97, 0.97);
  COVID C vs HAR-X +0.31 (0.13, 0.36) on 43 weeks; C vs unmanaged +0.72
  (0.33, 0.46); high-VIX C vs HAR-X +0.05 (0.43, 0.45). Nothing significant.
  Table 12 written (`results/tables/table12_sharpe_tests.tex`).
- Verdict: INCONCLUSIVE for the COVID headline, KEEP as a reported result;
  the paper may describe the COVID Sharpe gap but not call it evidence
  (finding #8).
- Cost: seconds.
- Lesson: 43 weekly observations cannot separate Sharpe ratios of 1.36 and
  1.06; say so rather than lead with the number.

## 2026-09-14  run-03  Do the pre-specified HAR-X-nesting specifications beat HAR-X?
- Setup: A1cs, A1sec, A1mod at h = 1, 5, 22 under the embargo (protocol steps
  3-5); `python -m modules.module4_benchmarks --only A1cs A1sec A1mod`, then
  `module12_incremental_tests.part_a`; OLS, no seed.
- Result (pooled, point-in-time). % change in MSE vs HAR-X / HLN-DM t /
  Clark-West t / CW p Holm-adjusted:
  A1cs   h1 -0.15% / -0.38 / +3.22 / 0.005;  h5 -0.18% / -0.37 / +3.36 / 0.004;  h22 -1.66% / -1.67 / +1.08 / 0.41
  A1sec  h1 -0.19% / -0.63 / +2.20 / 0.069;  h5 -0.38% / -1.07 / +1.97 / 0.099;  h22 -1.42% / -2.62 / +0.50 / 0.41
  A1mod  h1 -0.61% / -1.13 / +2.96 / 0.009;  h5 -0.60% / -0.80 / +3.12 / 0.006;  h22 -2.25% / -2.07 / +1.10 / 0.41
  C      h1 -0.59% / -0.74 / +6.12 / <0.001; h5 +0.57% / +0.48 / +7.74 / <0.001; h22 -3.53% / -2.43 / +6.26 / <0.001
  External (regimes, descriptive): C minus HAR-X at h = 5 is +1.9 pp in
  high-VIX weeks (DM t = 0.50) and -1.8 pp in COVID; A1mod h = 5 COVID
  +2.8 pp (DM t = 0.60). Every h = 22 regime difference is negative.
  QLIKE DM vs HAR-X: A1cs h5 +2.52 is the only |t| > 2 (unadjusted).
- Verdict: REVERT all three (decision rule: Holm CW p < 0.05 AND positive MSE
  gain; none satisfies both). C at h = 5 is the only row that satisfies the
  rule, with a +0.57% gain and DM t = +0.48 (finding #7).
- Cost: ~7 min of expanding OLS for 9 model-horizon fits.
- Lesson: Clark-West rejecting while MSE does not improve means the added
  regressors carry population information that per-stock OLS cannot turn
  into accuracy; the estimation noise of 3-13 extra coefficients per stock
  eats the gain. Pooled-panel estimation is the natural next hypothesis and
  is NOT part of this pre-registration.

## 2026-09-14  run-02  Point-in-time re-run of the ladder at h = 22
- Setup: embargo applied (protocol step 1); all linear models and the five
  ML estimators re-run at h = 22; Model C at h = 5 as a reproduction check.
  `python -m modules.module4_benchmarks --only A A1 A2 A3 A4 A5 C --horizons 22`,
  `--only C --horizons 5`, `python -m modules.module5_ml_models --horizons 22`.
- Result: C at h = 5 reproduces the saved file (max |diff| 9e-11, the
  float noise from the GPH rescaling). h = 22 pooled MSE, leaky -> point-in-time,
  gain vs HAR in brackets: A 0.2689 -> 0.2709; A1 0.2589 -> 0.2633 [+3.72% -> +2.82%,
  HLN-DM vs HAR t = 1.98 -> 1.49]; A2 -> 0.2760 [-1.89%]; A3 -> 0.2708 [+0.05%];
  A4 -> 0.2714 [-0.16%]; A5 -> 0.2699 [+0.39%]; C 0.2540 -> 0.2726 [+5.54% -> -0.62%];
  D_lasso -> 0.2703 [+0.22%]; D_ridge -> 0.2723 [-0.52%]; D_en -> 0.2704 [+0.19%];
  D_rf 0.2831 -> 0.2892 [-5.29% -> -6.75%, HLN-t -2.81]; D_gbm 0.3206 -> 0.3380
  [-19.22% -> -24.77%, HLN-t -6.42]. Tables 5-8 and Figures 4, 5, 8 regenerated
  from these files (module 6 then module 10; Figures 5 and 8 read module 6's
  table5_raw.csv / table8_raw.csv). Regime table at h = 22, C minus HAR-X:
  COVID -6.3 pp, high-VIX -4.5 pp, low-VIX +0.1 pp.
- Verdict: KEEP (these are the live h = 22 numbers; Tables 5-8 and Figures
  4, 7 regenerate from them). Findings #2, #6.
- Cost: linear ~12 min; ML h = 22 ~1.5 h (RF and GBM dominate).
- Lesson: at the monthly horizon nothing beats HAR-X and HAR-X itself is no
  longer significantly better than HAR.

## 2026-09-14  run-01  Direct tests of Model C against HAR-X on the saved (leaky) forecasts
- Setup: no model change; `diebold_mariano(loss_A1, loss_C)` from module 6 on
  the forecasts saved 2026-05; also the purge diagnostic (scratch script,
  embargo 0..4 rows at h = 22).
- Result: pooled MSE HLN-t vs HAR-X = -0.74 / +0.48 / +1.28 at h = 1/5/22
  (p = 0.46 / 0.64 / 0.20); QLIKE +0.54 / +1.20 / +1.65. Regimes: only
  low-VIX h = 22 significant (t = 3.57). Purge at h = 22: MSE(C) 0.2540 ->
  0.2726 as embargo goes 0 -> 4 rows (0.2540, 0.2628, 0.2679, 0.2709, 0.2726);
  embargo 0 reproduces the saved file to 1e-15. Purged h = 22: C vs HAR
  -0.62%, C vs HAR-X HLN-t = -2.43.
- Verdict: REVERT the published h = 22 results (finding #1, #2). The h = 5
  result stands (finding #3).
- Cost: ~12 min of expanding OLS.
- Lesson: an expanding window with overlapping multi-step targets needs an
  explicit embargo; the larger the model, the more it exploits the leak.
