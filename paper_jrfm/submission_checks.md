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

## Status (Akash, 2026-09-14, evening): rewritten around the corrected evidence; ready for Rachev's check

Rachev's 13 Sept items and what was done (full record: `docs/EXPERIMENTS.md` runs 01-09,
`docs/FINDINGS.md` #1-#12, `PROJECT_LOG.md`):

1. **Direct DM/HLN tests of C vs HAR-X at each horizon.** Done: new Section 8.3
   "Incremental Accuracy Relative to HAR-X" with the panel-aware HLN-DM and the
   Clark-West (2007) nested-model test for every specification, Holm-adjusted over the
   pre-specified comparisons (Table harx_tests), plus three follow-up designs fixed in
   advance (pooled estimation, term-structure target, market-level target; Table
   candidates). Result: Model C is never significantly better than HAR-X; the
   persistence information is real (Clark-West) but economically negligible.
2. **Point-in-time design.** A training-window leak at h = 22 was found and fixed
   (embargo in modules 4, 5, 9, 13; `tests/test_embargo.py`); winsorisation is now
   expanding-window; the portfolio normalisation constant is real-time (52-week
   warm-up) with the full-sample version kept as a robustness row; timing conventions
   (HAR components and rolling estimates end at t-1, VIX/MOVE at t) are stated in 7.6.
   The monthly-horizon gains in the 12 Sept draft were the leak; they are gone.
3. **S&P 500 membership.** Section 7.1 now states that the universe is the April 2026
   membership with no point-in-time reconstruction, and what that implies.
4. **Proofreading.** Eq. 13 punctuation, Bennedsen year (2022), new references
   (Clark-West 2007, Holm 1979, Ledoit-Wolf 2008, plus the six added on 13 Sept),
   Figure 8 caption, all numbers re-derived from the regenerated tables.

Compiled PDF: `Memory_Roughness_edited.pdf` (XeTeX build via Tectonic with the pdftex
option removed and the EPS logos swapped for their PDF versions; Nicholas should rebuild
with pdfTeX on Overleaf before sending). Corresponding author unchanged (Nicholas).

The framing has changed: the paper is now an honest characterisation (persistence is a
coherent descriptive market state whose forecasting content is already in implied
volatility). Rachev has not yet seen the GPH correction (13 Sept) or any of the above.
