# Findings

Numbered facts about the data, the metric, or the method, each traceable to a
ledger entry in [EXPERIMENTS.md](EXPERIMENTS.md). Overturned findings are
struck through with the date and run that overturned them.

#1 (2026-09-14): The expanding-window forecasts trained on rows whose target
window extended past the forecast origin. With a 5-day stride this affects
only h = 22 (last 4 training rows, up to 17 days of future variance). h = 1
and h = 5 are unaffected. From run-01.

#2 (2026-09-14): Model C's published monthly-horizon gain (+5.54% vs HAR,
+1.82 pp vs HAR-X, "largest in stress") was produced by that leak. Point-in-time,
C is -0.62% vs HAR at h = 22 and significantly worse than HAR-X (HLN-t = -2.43);
C minus HAR-X is -4.5 pp in high-VIX weeks and -6.3 pp in COVID. From run-01.

#3 (2026-09-14): At h = 5 (clean) Model C beats HAR by 8.24% (HLN-t = 3.87) but
beats HAR-X by only 0.52 pp, HLN-t = +0.48. At no horizon is C significantly
different from HAR-X on the saved forecasts (MSE or QLIKE). From run-01.

#4 (2026-09-13): The GPH estimator divided the slope by two; all GPH estimates
before 2026-09-13 were half their true value. Forecasts are invariant (features
scale by 2). From the module-2 re-run logged in PROJECT_LOG.

#5 (2026-04 to 2026-05, restated): Own-stock persistence and roughness (A2)
add nothing to HAR at any horizon; VIX and MOVE (HAR-X) account for most of
the pooled gain over HAR. From the Phase-7 ladder.

#6 (2026-09-14): Point-in-time, at h = 22 no specification beats HAR-X and
HAR-X itself is no longer significantly better than HAR (+2.82%, HLN-t = 1.49).
The monthly horizon carries no evidence for or against persistence. From run-02.

#7 (2026-09-14): Clark-West rejects the nested null against HAR-X for Model C
at every horizon (t = 6.1, 7.7, 6.3) and for the pre-registered HAR-X-plus-one-
block specifications at h = 1 and 5, yet no specification lowers pooled MSE
relative to HAR-X by more than 0.6% at any horizon, and none satisfies the
pre-registered rule (Holm CW p < 0.05 AND positive MSE gain) except C at h = 5
(+0.57%, DM t = 0.48). The persistence regressors carry population information
that per-stock OLS cannot convert into out-of-sample accuracy. From run-03.

#8 (2026-09-14): No volatility-managed Sharpe-ratio difference is significant.
COVID C-managed 1.36 vs HAR-X-managed 1.06: Ledoit-Wolf HAC p = 0.13,
block-bootstrap p = 0.36 (43 weeks). Full sample: +0.01, p = 0.62. From run-04.
Unchanged with real-time normalisation (COVID 1.41 vs 1.13, p = 0.15; full
sample 0.66 vs 0.63, p = 0.31). From run-09.

#9 (2026-09-14): Pooled estimation with stock fixed effects recovers only a
+0.20% MSE gain for the full persistence set over pooled HAR-X at h = 1
(Clark-West Holm p = 0.033, DM t = 0.36), +0.19% at h = 5 (Holm p = 0.058),
and a loss at h = 22. Pooling does not help HAR-X itself (+0.06 / -0.72 /
-0.01%). Estimation noise explains part of finding #7, but the recoverable
information is economically negligible. From run-05.

#10 (2026-09-14): The persistence state does not forecast the slope of the
future variance term structure (log ratio of monthly to weekly future mean
variance) beyond HAR-X: HAR-X + cross-sectional persistence -0.13%, full
set -1.80% (DM t = -3.12). The "duration of uncertainty" interpretation is
not supported as a forecastable quantity. From run-06.

~~#11 (2026-09-14): At the market level (cross-sectional mean of the stock
targets, one series) the persistence state adds nothing to VIX and MOVE:
-1.2% / -1.9% / -6.7% MSE at h = 1 / 5 / 22. Its information is the VIX
co-movement. From run-07.~~ Overturned 2026-10-02 by run-10: the result was
driven by the -17.5 log floor for missing stocks (audit I4) and the HAR timing
error (#13); corrected, the state adds +3.4% / +2.0% at h = 1 / 5 (DM 1.77 /
1.45, not significant). See #17.

#12 (2026-09-14): Point-in-time winsorisation and real-time portfolio
normalisation change no headline number by more than 0.01 pp (forecasts) or
alter any test conclusion (portfolios). From runs 08-09.

#13 (2026-10-02): HAR was fed variance through day t-1 while VIX/MOVE used the
close of day t. Correcting the timing lowers HAR's MSE by 8.1% (h=1), 7.0% (h=5),
2.9% (h=22) and HAR-X's gain over HAR from 5.10/7.73/2.82% to 3.11/5.54/1.77%.
About a third of the implied-volatility gain was a timing artefact. From run-10.

#14 (2026-10-02): The rolling persistence state is a window-inclusion indicator of
the largest recent spike: its COVID plateau ends exactly 500, 750, 1,000 trading
days after 18 Mar 2020 for those window lengths; max VIX in the trailing window
explains 79% of its variation (27% for current VIX); a short-memory AR(1)
(phi 0.90) with one year-long crisis episode reproduces the rise (0.19 -> 0.32)
and fall. From the audit diagnostics (shared audit doc N1, N2).

#15 (2026-10-02): The Hurst estimate of about 0.06 on daily log Parkinson variance
is reproduced (0.063 +/- 0.004) by a non-rough two-factor model plus range-estimator
noise; roughness is not identifiable from daily range data. From audit A6.
(Reproduced 2026-10-02 by run-17, module 16: 0.064 +/- 0.004 vs data 0.0605. The
noise-free latent model gives 0.19, so the low value comes from range noise and
from a fast mean-reverting factor, phi 0.66, together.)

#16 (2026-10-02): On excess returns every volatility-managed portfolio, the
model-free trailing-22-day-variance rule and equal weight have full-sample Sharpe
0.75-0.76; no difference is significant; managed books turn over about 10x a year.
Supersedes #8's framing. From run-10.

#17 (2026-10-02, exploratory): At the market level, crisis memory (max VIX over the
trailing three years) lowers the VIX's predictive slope for realised variance by
about 40% (interaction t = -2.7); this, not slow volatility levels, carries the
persistence state's market-level gain (+3.4% / +2.0% vs market HAR-X). Needs a
pre-registered out-of-sample test. From run-11. (Qualified 2026-10-02 by #19:
the slope effect is specific to the levels specification.)

#18 (2026-10-02): With hyperparameters tuned by time-series CV, gradient
boosting on Model C's predictors is -2.9 / -0.2 / -8.3% vs HAR (untuned:
-15.9 / -11.9 / -21.2%), level with the random forest, and both tree models
lose to HAR-X by 6-10% at every horizon (DM t -4.2 to -6.2). Shrinkage
estimators sit between HAR and HAR-X (Lasso -0.8 / -1.3 / -3.1% vs HAR-X).
Nonlinearity extracts nothing from this feature set; the earlier "gradient
boosting actively worsens forecasts" was an untuned-learner artefact. From
run-10 and run-12.

#19 (2026-10-02, exploratory): The crisis-memory effect of #17 is a levels
artefact plus a known premium. In log form the VIX x max-VIX interaction
vanishes (t +0.33 / +0.26); max-VIX-3y is an intercept shift that the trailing
gap between VIX-implied and realised variance absorbs, and that gap forecasts
better (+5.0 to +6.3% vs log HAR-X, DM 2.7 to 3.5). A persistent variance risk
premium after crises (Bates 2000; Andersen, Fusari & Todorov 2015), not a new
mechanism; effectively two episodes in sample. From run-13.

#20 (2026-10-02): Reproducible window-inclusion evidence (module 14). The
persistence state jumps when 18 Mar 2020 enters the window and drops when it
leaves, on 11 Mar 2022, 10 Mar 2023 and 8 Mar 2024 for W = 500, 750, 1000
(drops in the bottom 0.2% of all changes); max VIX in the window explains
81-83% of it, current VIX 18-28%. A short-memory ARMA(1,1) fitted to calm
data, with the 2020 episode injected, reproduces the calm level of d-hat
(0.40 vs 0.385), 53-69% of the rise and the exit at tau+W. A one-day spike
(Flash Crash) barely moves it: duration, not peak, drives the state. From
run-14. Supersedes the ad hoc evidence behind #14.

#21 (2026-10-05): A process WITH long memory (ARFIMA, d = 0.36 from calm
years) plus the 2020 episode produces the same entry-plateau-exit pattern as
the short-memory null (rise +0.11 vs +0.13 vs observed +0.19 at W = 750;
largest fall at day W + 9 for both). The rolling state's crisis signature is
therefore evidence about episodes passing through the window, not about
whether the baseline process has long memory; neither null reproduces the
post-exit level (data end below the pre-event level, nulls above). Max VIX in
the window remains the best single summary (R^2 0.83 vs 0.66 for the 90th
percentile). From run-24.

#22 (2026-10-05): The managed portfolios' Sharpe parity with equal weight
depends on same-close execution. Executing one trading day after the signal
lowers every managed Sharpe from 0.76 to 0.58 while equal weight moves from
0.76 to 0.72; managed portfolios are then 0.02 below equal weight (p about
0.26). Drawdown reduction (-0.16 vs -0.30) survives either timing. From
run-25.

#23 (2026-10-05): With a fixed 431-week estimation window the persistence
specifications lose more to HAR-X than with expanding windows (C: -2.9 /
-1.5 / -13.6% vs -1.0 / +0.3 / -3.4%), and every Giacomini-White rejection
goes against them. The expanding-window results are the persistence
specifications' best case. From run-19.
