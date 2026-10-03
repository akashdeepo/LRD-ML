# JRFM draft, October 2026

*Does Volatility Persistence Forecast Volatility? Rolling Long Memory, Crisis Windows, and Implied Volatility.* Deep, Appiah, Mei and Rachev.

- `main.tex`: front and back matter (MDPI class in `Definitions/`); one file per section in `sections/`.
- `make_tables.py`: builds every table in `tables/` and `figures/fig_cumulative.pdf` from `results/intermediate`, and copies Figure 1 from `results/figures`. Run it from the repo root after the pipeline: `python paper_jrfm_v3/make_tables.py`.
- `references.bib`: the September bibliography plus 34 entries fetched from doi.org.
- `main_draft.pdf`: a local XeTeX build (Tectonic, with the class's `pdftex` option removed and the EPS logos swapped for their PDF conversions). The canonical build is pdfTeX on Overleaf.

Every number in the text traces to a ledger run in `docs/EXPERIMENTS.md`: run-10 (corrected pipeline), run-12 (tuned gradient boosting), run-14 (window inclusion, Table 2 and Figure 1), run-15 (Giacomini-White), run-16 (timing, Table 6), run-17 (roughness, Appendix A).

## For the co-authors to confirm before submission

1. Author order, Hongwei Mei's affiliation, and the corresponding author (`main.tex`).
2. Author contributions: confirmed by Akash Deep on 2026-10-03.
3. The generative-AI statement that MDPI requires (`main.tex`, acknowledgments).
4. Whether to state in Appendix D that an earlier version (arXiv:2605.24285) reported larger gains, and that the corrections listed there explain the difference. Recommended, given the earlier similarity check.
5. The cover letter should disclose the arXiv preprint.
