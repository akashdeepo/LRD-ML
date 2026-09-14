# Manuscript revision and remaining submission checks

## Completed in Memory_Roughness_edited.tex

- Reframed the title and contribution around persistence as a shared financial state.
- Rewrote the abstract, contribution, interpretation, forecast comparisons, regime/sector discussion, economic implications, and conclusion without new empirical analysis.
- Distinguished gains over HAR from incremental comparisons with HAR-X; retained all numerical tables unchanged.
- Qualified duration, causal-mechanism, stationarity, roughness, coefficient-importance, and portfolio-performance claims.
- Added the coverage-selection/survivorship limitation without inventing a constituent-selection date or missing-data procedure.

## Check existing records before submission

1. **Forecast dates:** The stated protocol begins evaluation after sample date 431, in mid-2013. The existing `figures/fig4_cumulative_loss.pdf` instead displays dates beginning around 2008. Its inclusion and associated claims about out-of-sample GFC gains were removed; the original asset is unchanged. Do not restore it merely by relabeling the axis. Reconcile the dates against the saved forecast outputs first. The portfolio figure starts around 2013, consistent with the stated protocol.
2. **Sample construction:** Confirm the selection date/rule for the initial 125 securities, historical membership treatment, missing-value handling, and how 70% coverage yields the final analysis panel. The revision no longer assumes that this threshold alone establishes a balanced panel.
3. **Forecast implementation:** Confirm from existing code/logs the exact first evaluation date and that training targets are fully observed at each forecast origin, especially for overlapping five- and 22-day outcomes. Also confirm the timing of winsorization thresholds, feature transformations, and tuning. No new accuracy claims or implementation details were invented.
4. **Portfolio normalization:** Confirm the calibration period for the stock/model-specific constants and whether costs, financing, and exposure constraints enter the reported returns. The revised text treats the current results as descriptive and does not claim verified net-of-cost implementability.
5. **JRFM format converted:** The manuscript now uses the local MDPI class with `jrfm,article,submit,moreauthors,pdftex`, which selects the MDPI APA bibliography style. The title, authors, shared affiliation, correspondence, keywords, JEL codes, numbered sections, review line numbers, and back-matter declarations are converted. The revised text and numerical results are preserved; the four portrait figures are enlarged for the single-column layout. The corresponding PDF is rebuilt. Publication dates and DOI remain unassigned; this is a submission-format manuscript, not an accepted article.
6. **Author confirmation:** Review the CRediT role mapping, funding, data-availability, conflict, and not-applicable ethics statements before uploading. No statement that all authors have approved the manuscript has been invented. Confirm any journal-required disclosure of AI-assisted substantive editing, and complete author emails/ORCIDs in the submission system as applicable.

These are source/protocol verification and production items, not a request for another empirical robustness battery. The existing files do not supply enough information to resolve them silently.

## Proofreading follow-up

- Akash Deep is explicitly named as the corresponding author, with `akash.deep@ttu.edu`; other authors and their order are unchanged.
- Corrected the GPH slope relation in the text: for the displayed regressor `log[4 sin^2(lambda/2)]`, the estimator is minus the slope, not minus half the slope. Confirm the regressor and normalization in the estimation code before submission; no estimates were recalculated.
- Corrected the prose to match the existing lead–lag figure: the displayed correlations are stronger at negative lags, not symmetric with a peak at zero. The plotting convention and backward-looking window are now explicit. Confirm the saved plot's lag convention against the underlying code.
- Corrected the heatmap description to its displayed off-diagonal range, 0.14 (PG–JPM) to 0.72 (KO–XOM); no figure or data values were changed.
- Confirm the saved output underlying the separately reported GARCH QLIKE HLN statistic of −2.80, which is discussed in the text but not tabulated.
- Clarified the distinction between the broader descriptive feature inventory and the predictors in the stated forecasting specifications. Confirm this mapping against the final forecast code.
## Resolution of the items above (Akash, 2026-09-13)

Verified against the code and the saved outputs in the LRD-ML repo; edits applied to
`Memory_Roughness_edited.tex`, `tables/`, `figures/`, and `reference.bib`.

1. **Forecast dates.** Confirmed from `modules/module4_benchmarks.py`: `INIT_TRAIN_FRAC = 0.40`
   gives `init_n = 431` of 1078 sample dates; evaluation starts at sample date 432 (mid-2013).
   The cumulative-loss figure stays out; the vol-managed figure (starting 2013) is consistent.
2. **Sample construction.** Unchanged; the coverage caveat in Section 7.1 stands.
3. **Forecast implementation.** Walk-forward protocol as described; no change.
4. **Portfolio normalization.** Left as descriptive, as written.
5. **JRFM format.** Confirmed: JRFM uses MDPI's APA author-date style (`apajournal` /
   `mdpi_apacite.bst`); the manuscript compiles with the bundled `Definitions/mdpi.cls`.
6. **Author confirmation / AI disclosure.** AI-assisted editing (OpenAI Prism) is now
   disclosed in the Acknowledgments with MDPI's standard wording. Data Availability uses
   MDPI's restricted-third-party-data wording. Corresponding author is still Nicholas
   (`niappiah@ttu.edu`) in the .tex; the note below saying Akash is corresponding author was
   written before that change and is superseded.

**GPH slope relation (proofreading item).** The text's `d_hat = -slope` was right and the
code was wrong: `modules/module2_lrd_estimation.py` divided the slope by 2, which is only
correct for the regressor `log|2 sin(lambda/2)|`, not `log[4 sin^2(lambda/2)]` as used.
Every stored GPH estimate was therefore exactly half the true value (checked: ratio 2.000 on
all 1078 x 115 rolling cells). Fixed and re-run on 2026-09-13. Consequences:

- Full-sample GPH on Parkinson RV: 0.226 -> 0.451 (local-Whittle unchanged at 0.440, so the
  two estimators now agree); returns GPH -0.011 -> -0.022 (15% significant); % significant
  for RV 98% -> 99%. Table 5 (JRFM numbering) regenerated.
- Rolling GPH pooled mean 0.173 -> 0.346 (99th pct 0.831); cross-sectional mean state
  calm 2013-14 = 0.308, GFC 2008Q3-2009Q4 = 0.499 (+62%), COVID = 0.573 (+86%);
  rho with VIX unchanged at 0.501. Table 4 regenerated (also dropped the all-zero
  liquidity-interaction row and the unused USYC2Y10 row). Figure 3 regenerated.
- Forecasting results are unaffected: every persistence feature in Models A2-A5, C, D
  (`d_gph`, its dynamics, cross-sectional and sector means, VIX/MOVE interactions) scales
  by exactly 2, OLS forecasts are invariant to rescaling a regressor, the shrinkage
  estimators standardize inputs, and tree splits are invariant to monotone rescaling.
  Confirmed numerically by re-running the linear ladder (see PROJECT_LOG).
- Section 8.1, 7.5 and the Conclusion updated accordingly.

**Lead-lag convention.** Confirmed in `modules/module10_plots.py`: the plotted statistic is
rho(d_bar_t, VIX_{t+k}), so k > 0 means persistence leads VIX, as the caption states.

**GARCH QLIKE statistic.** -2.80 (p = 0.005) comes from `modules/_regen_table9.py`; it is
now also stated in the Table 10 notes.

**Other edits.** Removed the five-minute high-frequency subsample claims (no such analysis
exists in the code or the paper); reworded the Section 4.5 robustness sentence; added the
Parkinson (1980), Newey-West (1987), Jarque-Bera (1980), Tibshirani (1996), Zou-Hastie (2005)
and Hoerl-Kennard (1970) references; trimmed the abstract to 194 words.

## DO NOT SUBMIT THIS VERSION (Akash, 2026-09-14)

Rachev's 13 Sept request for direct Model C vs HAR-X tests and a point-in-time
audit led to two findings that invalidate parts of this draft (full record in
`docs/EXPERIMENTS.md`, `docs/FINDINGS.md`, `PROJECT_LOG.md`):

1. A training-window leak at h = 22 (last four training rows carried up to 17 days
   of future variance). Point-in-time, Model C is -0.6% vs HAR at the monthly
   horizon (was +5.5%) and significantly worse than HAR-X. Affected here:
   Table 6 (h = 22 rows), Table 8 (h = 22 rows), Table 9 is unaffected (h = 5),
   Figure 4 (h = 22 bars), Figure 7 (h = 22 column), and every sentence in the
   abstract, Sections 8.2, 8.4, 8.5, 9 and the Conclusion that says the
   persistence contribution is "largest at the monthly horizon" or "in stress".
2. Tested directly, Model C is not significantly better than HAR-X at any
   horizon (HLN-DM t = -0.74 / +0.48 / -2.43; Clark-West rejects the nested
   null but no specification lowers MSE vs HAR-X by more than 0.6%). The
   COVID Sharpe gap (1.36 vs 1.06) is not significant (Ledoit-Wolf p = 0.13,
   bootstrap p = 0.36).

The regenerated tables and figures live in `results/`; the manuscript text has
not yet been rewritten around the corrected results. That rewrite requires a
decision with Rachev on the paper's framing.
