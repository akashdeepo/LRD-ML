# JRFM draft, October 2026

*Does Volatility Persistence Forecast Volatility? Rolling Long Memory, Crisis Windows, and Implied Volatility.* Deep, Appiah, Mei and Rachev.

- `main.tex`: front and back matter (MDPI class in `Definitions/`); one file per section in `sections/`.
- `make_tables.py`: builds every table in `tables/` and `figures/fig_cumulative.pdf` from `results/intermediate`, and copies Figure 1 from `results/figures`. Run it from the repo root after the pipeline: `python paper_jrfm_v3/make_tables.py`.
- `references.bib`: the September bibliography plus 34 entries fetched from doi.org.
- `main_draft.pdf`: a local XeTeX build (Tectonic, with the class's `pdftex` option removed and the EPS logos swapped for their PDF conversions). The canonical build is pdfTeX on Overleaf.

Every number in the text traces to a ledger run in `docs/EXPERIMENTS.md`: run-10 (corrected pipeline), run-12 (tuned gradient boosting), run-14 (window inclusion, Table 2 and Figure 1), run-15 (Giacomini-White), run-16 (timing, Table 6), run-17 (roughness, Appendix A).