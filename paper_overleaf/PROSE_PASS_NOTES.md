# Prose Pass Notes for Nicholas

**Goal:** apply Rachev's 2 May polish list to `Memory_Roughness_Persistence.tex` before the final pre-arXiv check.

**Status:** technical revisions are done. This doc tells you what's already in the manuscript so you don't accidentally remove or contradict it during prose tightening.

---

## Rachev's checklist — what's left for you

| # | Rachev's request | Where in paper | Notes |
|---|---|---|---|
| 1 | Sharpen positioning in Introduction | end of §1, around line 207 | Add 3–4 explicit bullet/sentence statements. See "Bullet content" below. |
| 2 | New subsection: "Why persistence adds beyond HAR-X" | §6 (Machine Learning) or §8.2, your call | The framing he wants: HAR captures *time aggregation*, VIX captures *level of stress*, persistence captures *duration of stress*. See "HAR-X distinction" below. |
| 3 | Streamline §5 (~15% shorter) | §5 Structural Interpretation | Remove repetition. Add the formal sentence: "*$d_t$ is a reduced-form measure of the expected duration of volatility shocks.*" |
| 4 | Streamline §6 (ML) | §6 Machine Learning | ML is supporting, not central. Tighten algorithm descriptions; emphasize feature design. |
| 5 | Tighten abstract (~10%) | §abstract | See "Abstract priorities" below. |
| 6 | Interpretive sentence in each figure caption | all figures | One short sentence telling the reader the takeaway. See "Caption suggestions" below. |
| 7 | (Optional) Shorter title | top of file | Rachev's suggestion: *"Memory, Roughness, and Information Persistence in Financial Markets: A Structural Approach to Volatility Forecasting"* (drops "and Machine Learning"). Your call. |

---

## What's NEW in v2 you should preserve

### 1. Layered ladder $A \to A_1 \dots A_5 \to C \to D$
The model definition in §6 is the *layered* ladder, not the old $A/B/C/D$. $A_2$ is what was previously called Model B. Don't accidentally revert.

### 2. HLN-corrected Diebold–Mariano
All DM-$t$ values throughout the paper are the **panel-aware Harvey–Leybourne–Newbold** version (cross-sectional mean loss differential per date, Newey–West HAC, Student-$t(T-1)$). The old paper had iid pooled-cell DM-$t$ values around 15–24; the new HLN values are around 2–4. The shrinkage is real and correct — don't let "smaller looking" t-stats tempt you to revert.

Headline numbers: Model $C$ vs $A$ at $h \in \{1, 5, 22\}$ → HLN-DM-$t$ = $+3.15$, $+3.87$, $+2.33$. Significant at 1% / 1% / 5%.

### 3. GARCH(1,1) benchmark with QLIKE refinement
§8.6 contains a paragraph about GARCH(1,1) as an external benchmark. Two important points to preserve:
- GARCH MSE 0.703 vs HAR's 0.363 → MSE-HLN-DM-$t = -20.30$
- BUT under QLIKE (proxy-robust per Patton 2011) the gap is only HLN-DM-$t = -2.80$
- **Both numbers must stay** — the QLIKE result is what makes the comparison first-principles correct. Without it, the −94% MSE looks misleading.

### 4. Vol-managed portfolio (§9)
Moreira–Muir 2017 exercise. Headline: $C$-managed Sharpe in COVID = 1.37 vs unmanaged 0.65; CER (γ=5) = +12.3% vs −3.0%. Don't trim §9 too aggressively — the economic significance argument relies on this.

### 5. Robustness Table 9
12 rows including window 500 / 1000, alternative estimator (LW), alternative target (squared returns), liquidity halves, GARCH benchmark, regime rows. All confirm the headline.

### 6. Terminology
"**Parkinson range-based variance proxy**" — not "Parkinson realised variance". This swap is done globally; don't reintroduce "realised variance" except when referring to actual 5-min intraday RV.

---

## Bullet content for Introduction (item 1)

Rachev wants this as 3–4 explicit statements at the end of §1:

1. *Structural interpretation of persistence as a state variable* (the core conceptual contribution).
2. *Joint long-memory and rough-volatility empirical evidence on a 115-stock 25-year panel* (cross-sectional GPH $\hat d = 0.226$, Hurst $H = 0.063$).
3. *Layered ablation framework* that separates HAR vs HAR-X vs persistence aggregates.
4. *Economic significance via volatility-managed portfolios*.

---

## "HAR-X distinction" subsection (item 2)

Rachev's exact framing:

> HAR captures *time aggregation*; VIX/MOVE capture the *level* of stress; persistence captures the *duration* of stress.

Suggested 2–3 paragraphs:
- HAR's daily/weekly/monthly RV components capture the time-aggregation channel: short-run shocks die fast, long-run shocks die slowly.
- VIX and MOVE add a level dimension: how *intense* is current stress?
- Cross-sectional and sectoral $\bar d_t$, plus $\hat d \times$ VIX interactions, add a duration dimension: how *long-lived* will the consequences of current shocks be?
- Empirically the duration channel matters most at long horizons (h=22) and in stress regimes (high-VIX quartile, COVID), exactly where the HAR + level structure is least informative on its own.

---

## Abstract priorities

The current abstract is honest but a bit long. Rachev wants ~10% shorter. Priorities:

**Keep:**
- Joint long-memory + rough-volatility finding ($\hat d = 0.226 / 0.440$, $H = 0.063$)
- Cross-sectional mean rises 68% / 86% in GFC/COVID, $\rho = +0.50$ with VIX
- Headline forecast result: 4.6–8.2% MSE reduction, all significant under HLN-corrected DM
- Layered ablation finding: HAR-X captures most, persistence aggregates add incrementally at $h=22$ and in stress
- Vol-managed Sharpe in COVID: 1.37 vs 0.65
- Tree ML failure (RF flat, GBM negative)

**Trim:**
- The HLN methodology phrase is long; consider just "all significant under panel-aware DM inference."
- The HAR-X numbers in the abstract (+1.8 pp etc.) can be cut if space is tight; they're in the body.
- "Demonstrating economic significance beyond statistical gains" — short version.

---

## Caption interpretive sentences (item 6)

Suggested one-liners, one per figure:

| Fig | Suggested takeaway sentence |
|---|---|
| 1 | *Volatility clusters; squared returns and $\RVPK$ exhibit slow autocorrelation decay while raw returns do not.* |
| 2 | *The cross-sectional mean memory parameter rises sharply in the GFC and COVID and tracks the VIX.* |
| diag | *Returns are heavily fat-tailed; FIGARCH reproduces $\RVPK$-scale volatility well across the sample.* |
| 4 | *Model $C$ accumulates its advantage steadily across the OOS sample, with conspicuous jumps in the GFC and COVID.* |
| 5 | *Most of Model $C$'s pooled gain is captured by HAR-X ($A_1$); persistence aggregates add incremental value primarily at $h=22$.* |
| 6 | *Model $A_1$ beats HAR on 99% of stocks at $h=5$; Model $C$ on 94%; tree ML fails on the vast majority.* |
| 7 | *Persistence and market stress co-move primarily contemporaneously; no systematic lead/lag is detected (subject to the 750-day window smoothing).* |
| 8 | *Balance-sheet-sensitive sectors gain most from persistence features; tech and energy gain least, consistent with their idiosyncratic / single-factor exposures.* |
| 9 | *Volatility-managed portfolios cut exposure aggressively in COVID, sparing capital for the rebound; the persistence-augmented variant generates the largest Sharpe lift in stress.* |

---

## What NOT to remove during streamlining

- The Patton 2011 citation in §6 methodology block (governs QLIKE robustness).
- The Patton 2011 citation in §8.6 GARCH paragraph.
- The HLN-corrected DM methodology paragraph at the end of §6 (Newey–West HAC, $\lceil h/5\rceil-1$ bandwidth, Student-$t(T-1)$ reference).
- The Moreira–Muir 2017 citation in §9.
- The $A_2 \equiv$ legacy $B$ explanation (it's implicit in §6's ladder definition).
- The "level mismatch + GARCH limitation" two-cause explanation in §8.6 — without it the −94% MSE looks misleading.

---

## When you're done

1. Recompile in Overleaf, eyeball figures and tables.
2. Send PDF to me — I'll do a final technical sanity pass (numbers, cross-references, citations).
3. Send to Rachev for his pre-arXiv check.
4. Post on arXiv.
5. Submit to *Journal of Empirical Finance*.

— Akash
