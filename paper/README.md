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

## ⚠️ Before submission — three things that are NOT done

### 1. Complete the recent references

`refs.bib` opens with a block marked **RECENT / VERIFY**. Those are 2025–2026
works whose full author lists were not available offline, so they carry the first
author, `and others`, and the arXiv identifier taken from `ANALYSIS.md`. Open each
PDF and complete them:

| key | what to fill in |
|---|---|
| `segearthov3` | full author list, arXiv:2512.08730 — **our baseline, get this right** |
| `coninfer` | full author list, arXiv:2603.29271 |
| `sam3` | full author list and the arXiv number |
| `segearthov` | full author list, CVPR 2025 |
| `ovrsisbench` | authors and title, arXiv:2604.15652 |

**Do not guess an author list.** A wrong one is the error a reviewer who works in
the area spots immediately, and it costs more credibility than a missing citation.

### 2. Move to the venue template

`main.tex` compiles as-is against plain `article`, deliberately — it builds
anywhere, with no class file to chase. Two lines are marked `% TEMPLATE`:

```latex
\documentclass[10pt,a4paper]{article}         % TEMPLATE
\usepackage[margin=2.2cm]{geometry}           % TEMPLATE
```

For a CVPR workshop, replace both with the CVPR class (Overleaf carries the
template; start from it and paste the body in). For IEEE GRSL, use `IEEEtran`
with `journal` options. **Nothing else in the file assumes a document class** —
no hard-coded column widths, no `\linewidth` gymnastics — except that the two
wide figures use `figure*`, which is correct in two-column and harmless in one.

### 3. Length — about 2,000 words have to MOVE, not shrink

~7,100 words, 5 tables, 4 figures. That is roughly 10--11 two-column pages; a
CVPR workshop paper is **8 pages excluding references**, or about 5,000--5,500
words once floats are placed.

⚠️ **Word-level editing will not close this.** A full tightening pass over the
introduction and related work bought **69 words**. The prose is already dense, and
squeezing further starts costing clarity rather than length. Two blocks are marked
in `main.tex` instead:

```
% ==== SUPPLEMENTARY CANDIDATE 1: the refuted co-occurrence prior (~200 words)
% ==== SUPPLEMENTARY CANDIDATE 2: domain transfer, Table 4 + prose (~520 words)
```

The test applied to both: **is anything here load-bearing for a claim made in the
abstract?** Neither is. Moving both, plus their table, is roughly 2 pages.

If that is still not enough, the next candidates in order are the composition
decomposition (Section~\ref{sec:results}), then Figure~4, then the anatomy
section. Supplementary material is normal at CVPR workshops and costs nothing.

### The old cut list, retained --- and the principle

Currently ~7,200 words, 6 tables, 5 figures. A CVPR workshop paper is **8 pages
two-column excluding references**, which is roughly 5,000–5,500 words once
figures are placed. So **about 1,500–2,000 words have to go.**

Cut in this order, and note the principle: **cut a table, not a caveat.** The
caveats are what make the causal claim credible; the tables can move.

1. **Table 3 (co-occurrence ablation)** → one sentence. It is a refuted idea we
   keep for honesty, and one sentence discharges that duty.
2. **Table 5 (threshold rules)** → merge into the bound section as prose.
3. **Section 6 (what we built)** → compress. The atomisation ceiling is a
   two-sentence result; the prior's construction details belong in supplementary.
4. **Figure 5 (atom purity)** → supplementary. It supports a component that did
   not earn its place.

**Do not cut:** the intervention control arms, the saturation bound, the P8
failure, the units note, or the per-class decomposition of OpenEarthMap. Every
one of them exists because a reviewer would otherwise construct it themselves,
and the paper's credibility rests on getting there first.
