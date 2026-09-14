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
