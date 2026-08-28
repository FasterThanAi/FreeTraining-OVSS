# `paper/` — the LaTeX skeleton

Started at Week 3 rather than ROADMAP's Week 9, because writing is what reveals a
missing experiment while there is still time to run it.

## Build

No LaTeX is installed on the Mac. Use Overleaf, or any TeX Live:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are pulled from `../docs/` via `\graphicspath`. On Overleaf, upload
`docs/*.pdf` alongside `main.tex` and drop the `\graphicspath` line.

## ⚠️ `numbers.tex` is the only place a number may be typed

Every load-bearing figure is a macro carrying the results section it came from.
A number that is right in Table 1 and stale in the abstract is the cheapest
remaining class of error in this project, and macros make it impossible.

**If a number has no macro, it has no citation, and it does not go in the paper.**
To add one: put it in `numbers.tex` with its `WEEK*_RESULTS.md` section in a
comment, then use the macro.

## Section map

Follows `PAPER_OUTLINE.md` §3. Deviating from it means updating that file too —
it is the document that decides what each experiment is *for*.

## What `\todo{}` means

Unwritten prose, deliberately loud in red. The tables, figures and contribution
list are already real; the connecting text is not. Nothing marked `\todo` should
survive to submission.

## Template

Written against plain `article` so it compiles anywhere. Two lines marked
`% TEMPLATE` at the top switch it to CVPR (workshop, the primary target) or
IEEEtran (GRSL, the backup). Nothing else assumes a document class.
