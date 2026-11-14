# White paper (working draft)

- `whitepaper_src.html`: manuscript source. Aggregate numbers only; per-participant figures are referenced by file
  name and embedded at build time.
- `build.py`: embeds every figure found in `figs/` (a marked placeholder where a figure is missing) into
  `whitepaper.html`, then renders `whitepaper.pdf` with Playwright (`pdf.js`, needs node and the playwright package).

Figures: `scripts/whitepaper_sim_figures.py --out docs/whitepaper/figs` (simulations, no participant data) and
`scripts/whitepaper_figures.py` on the cluster (participant data, pseudonymized, written to outputs/whitepaper/).
`figs/`, `whitepaper.html` and `whitepaper.pdf` are never committed: once built they contain participant figures.
