# Project presentation

`main.tex` — Beamer, 16:9, 19 frames, following the committee's mandated
structure exactly (Introduction · Motivation & Applications · Problem Statement ·
Literature Survey · Research Gaps · Objectives · Experimental Setup & Results ·
References).

## Build

```bash
cd slides
latexmk -pdf main.tex          # or: pdflatex main; bibtex main; pdflatex main; pdflatex main
```

Needs `bibtex` — the reference slide is generated from `../paper/refs.bib`.

## ⛔ The rule that matters

**No number is typed in this file.** Every figure comes from `../paper/numbers.tex`,
the same macros the paper uses, so a slide cannot disagree with the paper. If a
number needs to change, change it there.

Images come from `../docs/` — the same PDFs the paper includes. Nothing is
re-rendered or re-cropped for the slides.

## Notes for the presenter

- **The contribution slide** is the one to land: not a new architecture, two
  lines of arithmetic, no gradient steps, and it works on a backbone it was not
  developed for.
- **The problem-statement slides** carry the mathematical formulation and the
  completeness proof. If the committee asks *"what is novel?"*, the answer is the
  monotone-equivalence argument: per-class τ is the **complete** family given a
  fixed argmax, so scaling before the argmax is provably a larger one.
- **Figure 8** — ⚠️ row four is a tile where the method *loses* 7.72 mIoU. Point
  at it before anyone else does; the five-fold gain is an average.
- **The limitations slide** — do not skip it. The transfer failure is the
  sharpest limitation, and stating it first is what makes the rest credible.
- 19 works cited against a minimum of 10.

⚠️ The survey slides summarise each work in one line. Read SegEarth-OV3,
ConInfer and SAM 3 properly before presenting — those three are what a committee
will actually ask about.
