# Paper — build and layout

This directory is **self-contained**: everything `main.tex` needs is inside it, so it can
be zipped and uploaded to Overleaf, arXiv, or a conference system without modification.

```
paper/
  main.tex        the paper
  refs.bib        bibliography (16 entries, all cited)
  REFERENCES.md   per-reference notes on the job each citation does — NOT part of the build
  figures/        vector PDFs, a copy of ../results/figures/
  README.md       this file
```

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Two `pdflatex` passes after `bibtex` are required: the first resolves citations, the
second the cross-references that shift once the bibliography changes their page numbers.

`latexmk -pdf main` does the same thing and stops when it converges.

## Overleaf

```bash
python scripts/make_overleaf_zip.py     # writes pemwe-paper-overleaf.zip
```

Then on overleaf.com: **New Project -> Upload Project -> select the zip**. Set the main
document to `main.tex` if it is not detected, and the compiler to **pdfLaTeX** (Menu ->
Compiler). Overleaf runs `bibtex` automatically, so citations resolve on the second build
— press Recompile twice if any show as `[?]` the first time.

`IEEEtran.cls` ships with Overleaf and is not vendored here. If you build somewhere
without it, fetch it from CTAN.

### Two things already handled, so you do not hit them

- **`\graphicspath` cannot use `../` on Overleaf** — there is no parent directory. The
  figures are copied into `figures/` and the path lists both locations.
- **`\day` and `\year` are not defined by siunitx v2**, and their availability has moved
  across v3 releases. A missing unit is a hard error, so both are declared explicitly in
  the preamble and the build no longer depends on which siunitx the host ships.

## Figures

`results/figures/` is canonical — regenerate with `python scripts/make_paper_figures.py`,
which also prints every performance number the Results section quotes.

`paper/figures/` is a copy that exists because a LaTeX project cannot reference a parent
directory on Overleaf. Refresh it and check for drift with:

```bash
python scripts/sync_paper_figures.py          # copy
python scripts/sync_paper_figures.py --check  # report only, exits 1 if stale
```

Run `--check` before submitting. A paper that silently builds against last week's figures
is a wrong paper that compiles cleanly.

## Open items

Four `\TODO` markers remain, all decisions rather than writing:

1. **Authors and affiliation.**
2. **Data provenance.** The profiles were built with the ERA5 fallback, not
   Renewables.ninja — check `data/processed/kutch_2019_1min.json`. If it says
   `open_meteo`, cite ERA5 and describe the PV and turbine conversions as ours. Only cite
   Pfenninger & Staffell if you re-run with a Renewables.ninja token.
3. **Hyperparameter table** in the Training subsection.
