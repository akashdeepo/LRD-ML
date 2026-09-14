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

#11 (2026-09-14): At the market level (cross-sectional mean of the stock
targets, one series) the persistence state adds nothing to VIX and MOVE:
-1.2% / -1.9% / -6.7% MSE at h = 1 / 5 / 22. Its information is the VIX
co-movement. From run-07.

#12 (2026-09-14): Point-in-time winsorisation and real-time portfolio
normalisation change no headline number by more than 0.01 pp (forecasts) or
alter any test conclusion (portfolios). From runs 08-09.
