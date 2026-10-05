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

**Pre-registered protocol, third block (written 2026-10-02, after the full
audit in the shared audit doc and before any corrected run).** The audit found
30 issues (3 critical, 14 major). This block fixes every one that the data at
hand allows and re-runs the whole pipeline once. Nothing else changes: same
sample, same specifications (A, A1..A5, C, D, A1cs, A1sec, A1mod, pooled,
duration, market level), same horizons, same decision rule (Holm-adjusted
Clark-West p < 0.05 AND positive MSE gain against the run's own HAR-X-type
baseline), DM reported as the stricter criterion.

Corrections (audit IDs):
- I1 timing. Every predictor uses information through the close of the
  origin day t: HAR terms are RV_t, mean RV_{t-4..t}, mean RV_{t-21..t};
  the return term is r_t; rolling d and H windows end on day t. VIX and MOVE
  are already day-t closes. Target unchanged (days t+1..t+h), embargo unchanged.
- I4 log floor: zero-range and missing days are NaN in log variance.
- I8 transform: GPH and local Whittle are estimated on log Parkinson variance
  throughout (headline, robustness, features), as is standard; Table 3 reports
  log-variance estimates.
- I10 regimes: one definition, VIX quartiles over the out-of-sample forecast
  dates, shared by every regime table; GFC rows dropped (no evaluation dates).
- I11 HAC bandwidth: max(ceil(h/5)-1, floor(4(T/100)^(2/9))) for DM and CW
  (6 lags at T of about 645); the short bandwidth is kept as a sensitivity column.
- I12 QLIKE: variance forecasts are smeared point-in-time (expanding mean of
  exp(past embargoed log errors), 26-error warm-up) for every model.
- I13 ML CV: TimeSeriesSplit(5, gap = ceil(h/5)) for Lasso, Ridge, Elastic Net.
- I14 GARCH: point-in-time level correction applied; recursion off-by-one fixed.
- I15 window robustness: same transform, H re-estimated, common cells.
- I20 sector mean is leave-one-out. I21 illiquidity interaction deleted.
  I22 missing d stays missing in threshold features.
- I23 every comparison on common (date, stock) cells; warm-up anchored on
  the date of sample row 431 for every model.
- I24 Clark-West only for OLS models that nest HAR-X (A5, A1cs, A1sec, A1mod, C).
- I25 duration design gains its no-regressor (expanding-mean) benchmark.
- I2, I5, I6, I26 portfolios: simple returns, excess returns over the 3-month
  T-bill (managed positions funded at the bill rate), common universe, ddof=1
  everywhere, a model-free benchmark scaled by trailing 22-day variance,
  turnover and Sharpe at 0, 5 and 10 bp per unit traded, leverage statistics.
- I7 data: KDP's 10 Jul 2018 special dividend patched; dividends otherwise
  disclosed (price returns). I18 High/Low ordered by max/min with an assertion.
- I3, I9: interpretive; disclosed (window-inclusion effect; roughness not
  identifiable from daily range data).
- I16, I17, I19, I28: text and stale numbers, carried into the 25-page draft.
- I27: tests for feature timing, NaN preservation, warm-up dates, DM against
  statsmodels.

Predictions written before the run: HAR improves materially (audit estimate
6 to 8% at h=1); HAR-X's gain over HAR shrinks by about a third; C versus
HAR-X stays insignificant at every horizon; managed-portfolio Sharpe
advantages over the unmanaged portfolio shrink to near zero on excess returns.

**Fourth block (written 2026-10-05, before any run; responds to the co-author
review of 2026-10-03, which all authors agreed to on 2026-10-04).** Nothing
here changes a registered specification. Each item adds a check, a matched
comparison or a sensitivity analysis, and its prediction is written here first.

- run-18 calendar. By horizon: last predictor date, last fully observed target
  date, candidate origins, exclusions, common cells; training counts under the
  embargo. No prediction (descriptive); known corrections: initial training
  ends 10 June 2013, h=22 uses 642 origins ending 16 March 2026, 1078 = 431 +
  646 + 1 (21 April 2026 has no target).
- run-19 Giacomini-White with bounded estimation memory. HAR-X, A1cs, A1sec,
  A1mod and C re-estimated with a fixed rolling window of 431 weekly rows
  (= initial training length, so the evaluation period is unchanged); GW on
  those forecasts with instruments (1, d_{t-k}); 260-row window as sensitivity;
  the conditional-moment regression coefficients reported with signs; a check
  that the instrument's target window has closed at the origin. Prediction: no
  GW rejection in favour of a persistence specification; rolling-window gains
  vs HAR-X within 1 percentage point of the expanding-window gains.
- run-20 Clark-West decomposition and size. For each Table 5 row the adjusted
  differential split into raw MSE differential and adjustment term, both HAC
  t-statistics. Monte Carlo size: 500 panels (115 series, 646 dates) with a
  common factor in errors (cross-sectional error correlation matched to the
  data) and per-stock nested OLS, 11 extra regressors with zero coefficients,
  same HAC bandwidth; nominal 5%. Prediction: the adjustment term exceeds the
  raw differential in every rejecting row; empirical size between 5 and 15%.
- run-21 bootstrap intervals. Circular block bootstrap over dates (block 12
  weeks, 5000 draws) for every relative MSE reduction in Tables 4, 5 and 7;
  95% percentile intervals. Prediction: every C-vs-HAR-X interval includes
  zero; the h=5 upper bound is below 1.5%.
- run-22 matched robustness. Table 8 Panel A variants with HAR-X refit on the
  same target, sample and cells; report C minus HAR-X with a bootstrap
  interval. Prediction: C vs HAR-X within [-1.5%, +1.5%] in every variant.
- run-23 memory-estimation checks. (a) Rolling GPH cross-sectional mean at
  bandwidth exponents 0.5, 0.6, 0.7 (W=750): correlation with the 0.65 series
  and the 2020 exit signature. (b) Validation test: GPH and local Whittle on
  ARFIMA(0,d,0), d in {0, 0.2, 0.4}, T=750, 200 replications; mean within 0.05
  of d. (c) Missing days: rolling GPH with linear interpolation of log variance
  over missing days vs dropping them; correlation of the two state series.
  Prediction: correlations above 0.95 in (a) and (c); (b) passes.
- run-24 window analysis additions. ARFIMA(0,d,0) null with d = median calm-
  period (2012-2019) local Whittle estimate, same innovation variance, same
  2020 episode injected, 200 paths; 5-95% bands for both nulls; all three
  events; R^2 of the state on trailing-W max VIX vs mean VIX, 90th percentile
  VIX, and episode length (days with VIX > 30 in the window). Prediction: the
  long-memory null also shows entry at tau and exit at tau+W; max VIX keeps the
  highest R^2 but the 90th percentile is within 0.05 of it.
- run-25 portfolios. Execution one day after the signal (weights from close t
  applied from close t+1); costs charged to equal weight too; leverage cap 2
  as sensitivity; T-bill conversion documented; leverage and exposure
  distributions reported. Total returns when the Bloomberg pull arrives.
  Prediction: full-sample Sharpe ratios change by less than 0.05; no
  significant difference appears.
- run-26 timing variants. HAR-X with VIX and MOVE lagged one day (strict 4 pm
  cutoff) vs aligned HAR; attribution to the joint VIX+MOVE block.
  Prediction: HAR-X's gain over HAR with lagged indices falls below the
  aligned gain by between 0.5 and 2 points at h=1.
- run-27 roughness sensitivity. Range-noise simulation at 390, 1560, 6240
  steps per day; theoretical variance of log(R^2) for the continuous range.
  Prediction: simulated H on the observed series moves by less than 0.01.
- run-28 sample record and balanced subsample. Selection trail from the raw
  pull (125 pulled, 115 kept, 107 with complete history); C vs HAR-X and HAR-X
  vs HAR on the 107-stock balanced panel. Prediction: gains within 0.5 points
  of the full-panel values.
- run-29 no-Hurst variant of C (16 predictors) at h=1, 5, 22 vs HAR-X.
  Prediction: within 0.3 points of C.
- run-30 (held for an overnight run) ML feature ablations: each learner on
  HAR-X's 7 predictors vs the 18, identical tuning, splits, refits and cells;
  scaling inside each CV fold; random forest tuned (LightGBM rf mode: leaves,
  min leaf, feature fraction by time-series CV). Prediction: for every learner
  the 18-predictor version is no better than the 7-predictor version at any
  horizon.

---

## 2026-10-05  run-23, run-24  Memory-estimation checks and window-analysis additions (module 20)
- Result run-24a (long-memory null: ARFIMA(0, 0.356, 0), d = median calm-period
  local Whittle estimate, innovation sd 0.732 matching the calm variance; same
  2020 episode; 200 paths): baseline / plateau / after-exit 0.365 / 0.474 /
  0.403 at W=750 (rise +0.108; ARMA null +0.130; observed +0.186); largest
  fall at day W+9 for every W, as for the ARMA null; 5-95% plateau band at
  W=750 [0.34, 0.59] (ARMA [0.41, 0.64]); observed plateau 0.571 lies inside
  both. Both nulls end above their pre-event level (ARFIMA +0.04, ARMA +0.07),
  whereas the data end below (0.327 vs 0.385): neither null reproduces the
  post-exit level.
- Result run-24b (R^2 of the W=750 state on window summaries of the VIX):
  max 0.83, 90th percentile 0.66, days above 30 0.64, mean 0.57, current 0.27.
- Result run-23a (bandwidth m = T^p at W=750, correlation with p=0.65; 2020
  entry / exit change): p=0.5 (m=27) 0.946, -0.04 / -0.07; p=0.6 (m=53) 0.993,
  +0.12 / -0.12; p=0.7 (m=102) 0.992, +0.13 / -0.10. Mean level falls from
  0.51 to 0.40 as m rises.
- Result run-23b (ARFIMA(0,d,0), T=750, 200 reps): GPH mean -0.003 / 0.205 /
  0.401 (sd 0.07-0.08), local Whittle -0.006 / 0.201 / 0.398 (sd 0.06-0.07)
  for d = 0 / 0.2 / 0.4.
- Result run-23c: inside-history missing days are 153 of 705,640 (144 zero
  ranges); all other gaps precede a stock's listing. Interpolating them leaves
  the state unchanged to three decimals (correlation 1.000).
- Verdict: KEEP. Predictions held except two: the 90th-percentile VIX is not
  within 0.05 of the maximum's R^2 (0.66 vs 0.83; the maximum is the best
  summary by a wide margin), and the p=0.5 bandwidth correlates 0.946, just
  below 0.95, with a much weaker 2020 signature (m=27 ordinates is too few).
  The main finding is unchanged: a process WITH long memory plus the episode
  produces the same entry-plateau-exit pattern, so the pattern is evidence
  about episodes, not about whether the baseline has long memory. The paper
  will say exactly that (review point 9). Finding #21.
- Cost: 25 minutes.
- Lesson: state the window length in trading days and the minimum number of
  valid observations (250); the missing-day question had a one-line answer.

## 2026-10-05  run-25  Portfolio implementation variants (module 21)
- Result (full-sample Sharpe, baseline / next-day execution / cap 2 / both):
  equal weight 0.76 / 0.72 / 0.76 / 0.72; HAR 0.75 / 0.59 / 0.74 / 0.59;
  HAR-X 0.76 / 0.58 / 0.76 / 0.58; trailing-22d 0.76 / 0.64 / 0.76 / 0.64;
  C 0.76 / 0.58 / 0.75 / 0.57. With next-day execution every managed
  portfolio is below equal weight by 0.02 (C vs EW: HAC p 0.26, bootstrap
  0.28) and C vs HAR-X is -0.001 (p 0.78). Maximum drawdowns are unchanged
  by execution timing (managed -0.15 to -0.17 vs EW -0.30). The cap of 2
  binds for the top 1% of positions and changes Sharpe ratios by at most
  0.006. Equal weight turns over 1.1x a year, so 10 bp costs 0.007 of Sharpe.
- Verdict: KEEP; prediction FAILED for execution timing. Moving execution to
  the day after the signal removes the managed portfolios' Sharpe advantage
  entirely (0.76 -> 0.58 while equal weight moves 0.76 -> 0.72): the
  variance signal's value is concentrated in the first day after the close.
  The paper's portfolio conclusion becomes: no detected Sharpe improvement
  under same-close execution, and a Sharpe disadvantage (not significant)
  under feasible next-day execution; drawdown reduction is robust to both.
  Finding #22.
- Cost: 3 minutes.
- Lesson: an execution lag of one day is not a detail for a volatility-timing
  strategy; report it as the headline implementation.

## 2026-10-05  run-26  Strict information cutoff (module 22)
- Result (HAR-X with VIX and MOVE lagged one trading day, vs aligned HAR):
  +2.18 / +4.15 / +1.25% (DM 4.71 / 4.67 / 0.67) against +3.11 / +5.54 /
  +1.77% with same-day indices; the same-day indices add 0.95 / 1.46 / 0.53%
  over the lagged ones (DM 3.45 / 4.06 / 1.84).
- Verdict: KEEP, prediction held (0.93 points at h=1, inside 0.5-2). Under
  the most conservative cutoff implied volatility still improves on HAR
  significantly at h=1 and 5; the paper reports both cutoffs and states the
  denominator: the stale-HAR gain (5.11%) exceeds the aligned gain (3.11%) by
  64%, i.e. 39% of the stale gain is timing.
- Cost: 3 minutes.

## 2026-10-05  run-27  Roughness discretization (module 22)
- Result: range-noise variance 0.356 / 0.342 / 0.334 and simulated H on the
  observed series 0.064 / 0.065 / 0.068 (sd 0.004) at 390 / 1560 / 6240
  steps per day, latent model fixed at the run-17 fit.
- Verdict: KEEP, prediction held (change < 0.01). The appendix reports the
  390-step value with the finer-grid range.
- Cost: 5 minutes.

---

## 2026-10-05  run-22, run-28, run-29  Matched robustness, balanced panel, no-Hurst variant (module 19)
- Result run-22 (C vs HAR-X at h=5, gain, DM t, 95% block-bootstrap interval):
  headline +0.26% (0.41) [-0.95, +1.39]; less liquid half +0.43% (0.62)
  [-0.90, +1.70]; more liquid half +0.08% (0.13) [-1.10, +1.22]; local Whittle
  +0.46% (0.63) [-1.01, +1.82]; window 500 -0.23% (-0.30) [-1.78, +1.32];
  window 1000 +0.69% (0.75) [-1.17, +2.33]; squared-return target, HAR-X
  refit on that target, -0.78% (-1.91) [-1.54, -0.04].
- Result run-28 (balanced 107-stock panel vs all 115): HAR-X vs HAR
  3.19 / 5.64 / 1.78% (was 3.11 / 5.54 / 1.77); C vs HAR-X -0.92 / +0.32 /
  -3.15% (was -1.00 / +0.26 / -3.37).
- Result run-29 (C without the two Hurst features, 16 predictors, vs HAR-X):
  -0.68 / +0.21 / -2.81% (DM -1.50 / 0.36 / -2.56) against C's -1.00 / +0.26
  / -3.37%.
- Verdict: KEEP. All predictions held (run-29 at h=22 differs by 0.55 points,
  above the 0.3 predicted, in the direction of a smaller loss). Table 8 Panel
  A is replaced by the matched comparison: no variant gives model C a
  significant gain over HAR-X; the squared-return target gives a significant
  loss at the 10% level.
- Cost: 50 minutes.
- Lesson: the matched comparison is the one the claim needs; "gain of the
  same order as HAR-X's" was not evidence about persistence.

---

## 2026-10-05  run-19  Giacomini-White with bounded estimation memory (module 18)
- Result (rolling window 431 weekly rows; gain vs rolling HAR-X, DM t, GW p,
  sign of the conditional-moment intercept): A1cs -0.38 / -0.50 / -4.61%
  (t -1.01 / -0.70 / -2.43; GW p 0.48 / 0.33 / 0.048); A1sec -0.08 / -0.15 /
  -2.85% (GW p 0.65 / 0.94 / 0.024); A1mod -1.08 / -1.46 / -7.38% (GW p
  0.085 / 0.12 / 0.010); C -2.88 / -1.46 / -13.57% (DM -4.80 / -1.58 / -4.97;
  GW p <0.001 / 0.16 / <0.001). Every GW rejection has a negative intercept
  (the persistence model loses on average) and no significant slope on the
  lagged differential except A1mod at h=22 (t 1.84). Instrument lag 5 / 5 / 25
  trading days, so the instrument's target window is closed at every origin.
  Window 260: all specifications worse than rolling HAR-X, C by -7.1 / -5.7 /
  -42.6% (18 predictors on 260 rows), A1mod -59.6% at h=22.
- Verdict: KEEP, prediction half-held. No GW rejection favours persistence
  (held). The gains are NOT within 1 point of the expanding-window gains for C
  (expanding -1.00 / +0.26 / -3.37%): with bounded memory the richer models
  lose more, so the expanding-window numbers are the persistence
  specifications' best case. Reported as such; the expanding-window GW
  p-values in Table 5 are kept with the caveat that their justification is
  the rolling-window version.
- Cost: 22 minutes.
- Lesson: the persistence specifications' small expanding-window losses rest
  on long estimation samples; shorten the window and estimation noise
  dominates.

---

## 2026-10-05  run-18, run-20, run-21  Calendar, Clark-West decomposition and size, bootstrap intervals (module 17)
- Result run-18 (results/intermediate/calendar_report.csv): 6136 trading days
  to 21 Apr 2026; 1078 weekly origins (22 Nov 2004 to 21 Apr 2026); initial
  training 431 origins ending 10 Jun 2013; evaluation from 17 Jun 2013 on 646
  origins to 14 Apr 2026 at h=1, 5 and 642 origins to 16 Mar 2026 at h=22;
  excluded for no observed target: 1 / 1 / 5 origins; common HAR/HAR-X cells
  74,290 / 74,290 / 73,830. Embargo at h=22 drops the last 4 training rows
  (JPM: 427 training rows at the first evaluation origin, 1068 at the last).
- Result run-20: in every Clark-West row the adjustment term exceeds the raw
  MSE differential (adjustment share 0.96 to 4.45); the raw differential is
  negative in 12 of 15 rows (HAC t from -2.71 to +0.41) while the adjustment
  term has t of 7.8 to 18.0. Monte Carlo size of the panel CW test under the
  null (common error factor, cross-sectional error correlation 0.216 as in the
  data, 11 extra zero-coefficient regressors, per-stock expanding OLS): 5.2%
  at nominal 5% (one-sided) at both bandwidth settings.
- Result run-21 (block 12, 5000 draws): C vs HAR-X 95% intervals h=1/5/22
  [-1.84, -0.17], [-0.93, +1.39], [-5.89, -0.97]; A1cs [-0.65, +0.52],
  [-0.90, +0.86], [-4.01, +0.10]; A1sec and A1mod include zero at h=1, 5 and
  are at or below zero at h=22; HAR-X vs HAR [2.19, 4.09], [3.87, 7.18],
  [-2.21, 5.23]; every ML row vs HAR-X has an interval entirely below zero.
- Verdict: KEEP. run-20 prediction held (adjustment dominates; size 5.2% is
  inside 5-15%). run-21 predictions held (C vs HAR-X includes zero only at
  h=5, upper bound 1.39 < 1.5). The CW rejections are therefore not a size
  problem: the test is correctly sized and rejects because the adjustment term
  (the squared forecast difference) is large, i.e. the extra coefficients are
  nonzero in population, while the realised loss does not improve.
- Cost: 4 minutes.
- Lesson: report CW together with its decomposition; on its own it reads as
  evidence of forecast improvement that the raw differential contradicts.

---

## 2026-10-02  run-17  Roughness null, made reproducible (module 16; for the JRFM appendix)
- Setup (written before the run): finding #15 (H ~ 0.06 reproduced by a
  non-rough model plus range noise) came from an ad hoc audit diagnostic.
  Module 16: latent log variance = two independent AR(1) components; observed
  = latent + eta, where eta = log(R^2 / (4 ln 2)) minus its mean and R is the
  range of a standard Brownian motion over a day sampled at 390 one-minute
  steps (the Parkinson sampling error; no free parameter). The four AR
  parameters are fitted by least squares to the median-across-stocks
  autocorrelations of observed log variance at lags 1, 2, 3, 5, 10, 22, 66,
  125, 250 and its variance, net of eta's variance. 200 simulated paths of
  6136 days; Hurst exponent by module 2's increment-scaling estimator (lags
  1-21, q = 2) on observed and on latent paths.
- Prediction: simulated observed H within 0.03 of the data's mean (0.060);
  latent H at least 0.2 higher than simulated observed H.
- Result: data H 0.0605 (10th-90th pct 0.053-0.069). Fitted latent model:
  AR(1) components phi 0.661 (share 0.37) and 0.9953 (share 0.63); range
  noise variance 0.355 of total 1.071; ACF fit SSE 0.0024 over nine lags.
  Simulated H: observed 0.064 (sd 0.004), latent 0.191 (sd 0.007).
- Verdict: prediction 1 held; prediction 2 FAILED (latent exceeds observed by
  0.13, not 0.2). A Markovian two-factor model reproduces H ~ 0.06 exactly,
  so roughness is not identified from daily ranges (#15 stands), but the low
  value has two sources: range noise (0.19 -> 0.06) and a fast mean-reverting
  factor that already pulls the increment-scaling estimator well below 0.5 at
  lags of 1-21 days. Finding #15 reworded accordingly; appendix A of the
  JRFM draft states both.
- Cost: about 12 minutes.
- Lesson: the increment-scaling Hurst estimator at daily lags is biased low by
  any fast mean reversion, not only by noise.

---

## 2026-10-02  run-16  Timing alignment isolated (module 15; for the JRFM paper)
- Setup (written before the run): finding #13 attributes HAR-X's shrinking
  gain over HAR (5.10 -> 3.11% at h=1) to the HAR timing fix, but run-10
  changed several things at once. Module 15 isolates it on the current
  pipeline: HAR and HAR-X re-estimated with HAR terms built from RV through
  t-1 and the return term r_{t-1} (VIX and MOVE stay at the close of t),
  everything else identical (same expanding OLS, embargo, common cells).
  Compared with the aligned A and A1 forecasts from run-10.
- Prediction: the stale alignment raises HAR's MSE by 6 to 9% at h=1 and
  raises HAR-X's gain over HAR by 1.5 to 2.5 points at h=1, less at h=5, 22.
- Result: stale vs aligned, h=1/5/22: HAR MSE 0.6691 / 0.3637 / 0.2712 vs
  0.6146 / 0.3381 / 0.2632 (aligned HAR better by 8.1 / 7.0 / 2.9%, DM t
  6.39 / 5.00 / 3.41); HAR-X gain over HAR 5.11 / 7.76 / 2.90% vs 3.11 /
  5.54 / 1.77%. The stale numbers reproduce the pre-audit ones (HAR 0.6687,
  HAR-X gain 5.10 / 7.73 / 2.82%), so the timing alone explains the change;
  the share of the apparent VIX gain due to timing is 39 / 29 / 39%.
- Verdict: KEEP. Prediction held for h=1 (HAR MSE +8.9% when stale; gain +2.0
  points); the h=5 effect (+2.2 points) is not smaller than h=1, as predicted,
  while h=22 is (+1.1). Finding #13 now rests on an isolated comparison.
  Table 15 for the JRFM paper.
- Cost: about 4 minutes on 12 processes.
- Lesson: isolate a correction before attributing a change to it; here the
  attribution was right, now it is shown.

---

## 2026-10-02  run-15  Giacomini-White tests and an economic margin (for the JRFM paper)
- Setup (written before the GW numbers were seen): module 12 adds the
  Giacomini-White (2006) conditional predictive ability test of every
  specification against HAR-X, on the cross-sectional mean loss differential,
  instruments (1, d_{t-k}) with k = ceil(h/5), chi-square(2), module-6 HAC
  bandwidth. Purpose: finding #7 (Clark-West rejects, DM does not) says the
  population model has information the estimated model cannot use; GW tests
  the estimated model directly. Also a reporting flag, adds_beyond_harx_margin
  = pre-registered rule AND gain >= 0.5% of HAR-X MSE. The margin is NOT
  pre-registered: it is added after every MSE gain is known (largest positive
  vs HAR-X: C +0.26% at h=5) and is labelled as such in the paper.
- Prediction: no specification has GW p < 0.05 together with a positive MSE
  gain over HAR-X; the GW rejections that occur favour HAR-X (C at h=1, 22).
- Result: GW p vs HAR-X, h=1/5/22: A1cs 0.74 / 0.92 / 0.13; A1sec 0.67 /
  0.54 / 0.42; A1mod 0.40 / 0.28 / 0.11; C 0.064 / 0.70 / 0.018 (C worse than
  HAR-X at h=1 and 22). Every ML and own-stock row rejects or is near
  rejection in HAR-X's favour. adds_beyond_harx_margin is False everywhere
  (the rule's two passes, C +0.26% and A1cs +0.003% at h=5, are below 0.5%).
- Verdict: KEEP. Prediction held for positive gains; half held for the
  rejections (C at h=22 rejects, h=1 is 0.064). Clark-West says the
  population model has information; GW says the estimated model never turns
  it into conditionally better forecasts. Table 11 gains a GW column.
- Cost: under a minute.
- Lesson: CW and GW answer different questions; report both when they can
  disagree.

---

## 2026-10-02  run-14  Window inclusion, made reproducible (module 14; for the JRFM paper)
- Setup (written before the run): the window-inclusion evidence (audit N1, N2)
  existed only as ad hoc diagnostics. Module 14 recomputes it from the panel.
  (1) Rolling GPH d-hat of log Parkinson variance at W = 500, 750, 1000 (stride
  5, window ends on day t, bandwidth T^0.65), cross-sectional mean d-bar_t(W).
  (2) Events, rule fixed now: the 3 largest days of the cross-sectional mean
  daily log variance, at least 250 trading days apart. For each event tau and
  W: entry change (mean d-bar over the 4 grid points with window end in
  [tau, tau+20) minus the 4 in [tau-20, tau)) and exit change (same around
  tau+W), each ranked among all such 4-vs-4 changes. (3) R^2 of d-bar_t(W) on
  max VIX over the same trailing W days vs on current VIX. (4) Matched
  short-memory null: per-stock ARMA(1,1) fitted to calm log variance
  (2012-2019), median parameters; inject the observed one-year COVID excess
  profile (cross-sectional median of log variance minus calm mean, from the
  first event day in 2020); 200 replications; rolling GPH at each W.
- Predictions: (a) for the 2020 event the exit change is negative and in the
  bottom 5% of all changes at every W; (b) R^2 on max VIX in the window exceeds
  R^2 on current VIX at every W; (c) the short-memory null shows a plateau that
  ends at tau+W, with a rise at least half the observed rise.
- Result: events 10 Oct 2008, 6 May 2010 (Flash Crash), 18 Mar 2020. 2020
  exit change at W = 500 / 750 / 1000: -0.164 / -0.108 / -0.090, percentile
  0.18 / 0.09 / 0.10, on 11 Mar 2022 / 10 Mar 2023 / 8 Mar 2024; entry
  +0.18 / +0.13 / +0.11 (99.7-99.9th pct). 2008 exit at W=750 and 1000 in the
  bottom 2% (at W=500 the long 2008-09 episode and May 2010 are still inside).
  The one-day 2010 event barely moves d-bar (entry 23rd-90th pct). R^2 on max
  VIX in the window 0.81 / 0.83 / 0.82 vs current VIX 0.28 / 0.27 / 0.18.
  Null (ARMA(1,1), calm medians phi 0.91, theta -0.69): baseline d-bar 0.41 /
  0.40 / 0.39 vs observed 0.35 / 0.385 / 0.37; rise +0.13 / +0.13 / +0.11 vs
  observed +0.25 / +0.19 / +0.16 (53-69%); largest simulated drop at day
  W+9 for every W.
- Verdict: KEEP. All three predictions held ((c) narrowly at W=500, 53%).
  Finding #20. Table 14 and Figure 10 for the JRFM paper.
- Cost: about 6 minutes.
- Lesson: the dose-response (long episodes move d-hat, one-day spikes do not)
  and the calm-level match came free with the design; both belong in the paper.

---

## 2026-10-02  run-13  EXPLORATORY: referee checks of the crisis-memory lead (not pre-registered)
- Setup: run by an independent referee review after run-11; re-run and confirmed
  by the main session. Scratch scripts referee_vix_form.py and
  referee_vix_mech.py (session scratchpad). Same market-level target and
  expanding OLS with embargo as run-11; adds a log-form specification (log VIX,
  log MOVE, log max-VIX-3y), a VIX-squared control, the trailing 250-day gap
  between VIX-implied log variance and realised average-stock log variance
  ("bias", a variance-risk-premium plus correlation proxy), a realised-
  correlation proxy, and the SPX close-to-close log variance target.
- Result: run-11 reproduced exactly (+3.73% / +3.27%, DM 1.85 / 1.62;
  interaction t -2.74 / -2.65). In log form the interaction vanishes
  (t +0.33 / +0.26 at h=1/5); crisis memory acts as an intercept shift
  (log max-VIX-3y coefficient about -0.15 to -0.18, t about -4). The trailing
  implied-minus-realised gap absorbs it (h=22: max-VIX t 0.0 with the gap) and
  forecasts better: +5.0 to +6.3% vs log HAR-X (DM 2.7 to 3.5), against
  +0.3 to +1.9% for max-VIX alone. Survives a realised-correlation proxy and
  holds for SPX close-to-close variance (h=5 coefficient -0.62, t -5.9; OOS
  +4.7%, DM 2.5). Max-VIX-3y is 80.9 or 82.7 across the top quartile of dates:
  effectively two episodes.
- Verdict: INCONCLUSIVE by construction (exploratory). The lead is a
  persistent variance risk premium after crises, which is known (Bates 2000;
  Chernov 2007; Andersen, Fusari & Todorov 2015), not a new slope effect.
  Finding #19 qualifies #17. Any claim needs held-out data (VXO 1986-2001 with
  SPX ranges; V2X, VDAX, Nikkei VI, VHSI, VKOSPI with own-index ranges) and a
  pre-registered log-form test controlling for the gap.
- Cost: seconds.
- Lesson: an interaction found in levels must be re-checked in the model's
  natural (log) form and against the obvious known mechanism before it is
  called a lead.

---

## 2026-10-02  run-12  Gradient Boosting tuned by time-series CV (audit I32)
- Setup (written before the run): the manuscript says hyperparameters are
  chosen by time-series cross-validation; in module 5 that holds for Lasso,
  Ridge and Elastic Net only. Gradient Boosting now chooses num_leaves in
  {2, 4, 15}, min_data_in_leaf in {20, 50} and the number of trees in
  {25, 50, 100, 200, 400, 800} (learning rate 0.05) at every refit by
  TimeSeriesSplit(3, gap = ceil(h/5)) on the embargoed training rows only,
  then refits on all of them. (Amended before the full run: the first grid,
  leaves {4, 15} and up to 400 trees, put AAPL's training-CV optimum on its
  edge, at stumps with 800 trees; for JPM, XOM, KO and NVDA 4 leaves was
  interior. The change rests on training-fold CV error; a code smoke test had
  scored the first grid out of sample on AAPL, JPM, XOM at h=5 only: MSE
  0.393 / 0.328 / 0.313 vs untuned 0.444 / 0.353 / 0.314 vs HAR 0.365 /
  0.343 / 0.278.) Everything else as run-10 (refit every 20
  origins, same embargo, same features as C). Random Forest keeps its fixed
  settings (200 trees, min leaf 20; robust defaults) and the text will say so.
  Module 6, 10, 12 re-run after. The untuned run-10 numbers stay in this ledger.
- Hypothesis: the run-10 Gradient Boosting losses (-15.9 / -11.9 / -21.2% vs
  HAR) are overfitting, not evidence about nonlinearity. Prediction: tuned
  Gradient Boosting lands between -5% and +5% of HAR at h=1 and h=5 and stays
  below HAR-X at every horizon, as the linear ML models do.
- Result (vs HAR, h=1/5/22; untuned run-10 in brackets): Gradient Boosting
  -2.88 / -0.23 / -8.32% (-15.91 / -11.91 / -21.15%), HLN-t -3.16 / -0.19 /
  -2.96. Vs HAR-X: -6.18 / -6.11 / -10.27%, DM t -6.22 / -5.38 / -4.19.
  Stocks beating HAR at h=5: 60 of 115 (was 7). Now level with Random Forest
  (-2.49 / -0.34 / -5.11%); no other row changed.
  Also rebuilt (audit I33): per_stock_improvement.csv, which feeds Figure 6
  and the per-stock shares in the text, had been written in May by an
  uncommitted script and never refreshed; module 6 now writes it on common
  cells. Stocks beating HAR at h=5: HAR-X 111 (May file 114), C 107 (108);
  at h=22 C 48 (94: the May file predates the leak fix).
- Verdict: KEEP. Both predictions held. The tree-model rows are now a fair
  test: with tuned hyperparameters neither tree model beats HAR at any
  horizon, and both lose to HAR-X by 6-10%. Finding #18.
- Cost: 80 min on 12 processes (about 32 min per horizon) plus 2 min evaluation.
- Lesson: an untuned learner is a strawman; a claim that a method "actively
  worsens forecasts" needs the method tuned under the same protocol as its
  competitors. And every figure input must have a committed writer.

---

## 2026-10-02  run-11  EXPLORATORY: what carries the market-level gain? (not pre-registered)
- Setup: market-level target (cross-sectional mean of stock targets), expanding
  OLS with embargo, ad hoc specifications run after seeing run-10's market
  result: HAR-X plus 66/250/750-day log-variance averages; plus the persistence
  state on top; plus max VIX over the trailing 750 days and its interaction with
  VIX; log-VIX variant. Scratch script; not in the repo pipeline.
- Result (gain vs market HAR-X, h=1 / h=5): slow averages -0.62% / -0.75%;
  state on top of slow averages +3.97% (DM 2.21) / +3.22% (DM 2.00); max-VIX-3y
  plus VIX x max-VIX-3y +3.73% (DM 1.85) / +3.27% (DM 1.62); persistence level
  alone +1.29% / +0.56%. Full-sample OLS: VIX x log max-VIX-3y coefficient
  -0.014 (t -2.74, h=1), -0.016 (t -2.65, h=5); VIX slope 0.038 at low crisis
  memory vs 0.022 at high (h=5).
- Verdict: INCONCLUSIVE by construction (exploratory, chosen after the result).
  Strong lead: crisis memory lowers the VIX's predictive slope for realised
  market variance by about 40%. Finding #17. Must be pre-registered and tested
  on held-out data (pre-2004 or other markets) before any claim.
- Cost: seconds.
- Lesson: the persistence state's only useful content is a noisy proxy for a
  simple crisis-memory variable.

## 2026-10-02  run-10  Corrected pipeline (audit fixes, third protocol block)
- Setup: every fix in the third protocol block; full re-run from the raw-panel
  rebuild (modules io_v2, 1, 1b, 2, 3, 4, 4b, 13, 9, 6, 10, 11, 12); ML
  (module 5) re-running detached, rows below marked pending. 19 tests pass.
- Result (before -> after). HAR MSE h=1/5/22: 0.6687->0.6146, 0.3634->0.3381,
  0.2710->0.2632. HAR-X vs HAR: +5.10->+3.11%, +7.73->+5.54%, +2.82->+1.77%
  (t 0.95 at h=22). C vs HAR: +4.55->+2.14%, +8.25->+5.79%, -0.62->-1.54%.
  C vs HAR-X: MSE -1.00 / +0.26 / -3.37%, DM -2.14 / +0.41 / -2.69.
  A1cs vs HAR-X +0.00% at h=5 (CW Holm p<0.001); A1sec, A1mod negative at all h.
  Pooled P-C vs P-A1 +0.40% / +0.34% / -1.36% (DM 1.19 / 0.79). Duration:
  C vs A1 -1.81% (DM -3.92); HAR beats the expanding mean by 3.35%. Market
  level M-C vs M-A1: +3.42% / +2.01% / -1.93% (DM 1.77 / 1.45), was -1.22 /
  -1.90 / -6.66. Robustness (C vs A, h=5): +5.3% to +6.4% under every variant;
  level-corrected GARCH -19.9%. Portfolios (excess returns, 594 weeks): Sharpe
  0.75-0.76 for every portfolio incl. equal weight and the model-free rule;
  COVID C 1.62, HAR-X 1.38, EW 0.88 (C vs HAR-X HAC p 0.10, bootstrap 0.31);
  turnover about 10x/yr managed vs 1.1x EW; Sharpe at 10 bp 0.66 vs 0.76.
  Persistence on log variance: GPH 0.538, LW 0.516 (84% of stocks > 0.5);
  crisis rise +38% (GFC) / +35% (COVID); max-VIX-in-window R2 0.79.
  ML (module 5, completed 19:06, all 15 forecast files refreshed): vs HAR
  h=1/5/22 Lasso +2.31 / +4.30 / -1.26%, Ridge +1.65 / +3.78 / -2.37%,
  Elastic Net +2.32 / +4.25 / -1.46%, Random Forest -2.49 / -0.34 / -5.11%,
  Gradient Boosting -15.91 / -11.91 / -21.15%. Every ML row is below HAR-X at
  every horizon, DM-significantly (Lasso -0.82 / -1.32 / -3.09%, DM -2.61 /
  -2.19 / -2.57). Gradient Boosting has fixed hyperparameters (400 trees,
  31 leaves, no early stopping) on 430-1,100 rows per stock: audit I32, run-12.
- Verdict: KEEP (these are the live numbers). All four pre-registered
  predictions held. The decision rule is met by C at h=5 (+0.26%), A1cs at h=5
  (+0.003%) and the pooled designs at h=1, 5, none DM-significant: the rule
  needs a minimum economic margin (open question for co-authors).
  Findings #13-#16; #8, #10, #11, #12 updated.
- Cost: linear ~25 min in four parallel jobs; candidates, GARCH, robustness
  ~10 min each; ML several hours (Random Forest dominates).
- Lesson: a one-day misalignment between HAR inputs and implied-volatility
  inputs inflated the implied-volatility gain by a third; align every
  predictor to the forecast origin and test it.

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
  changes. ML refresh completed 20:15: every D row moved by at most
  0.06 pp (e.g. GBM h=5 -9.86% -> -9.80%, Lasso h=22 +0.23% -> +0.21%);
  Tables 5-8 and 11 and Figures 4-8 regenerated and ported to the bundle.
- Cost: ~20 min linear; ML 6 h 19 min (Random Forest 78-91 min per
  horizon on 12 cores; scikit-learn forests have no GPU path).
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
