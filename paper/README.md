# The paper — two documents, one set of numbers

| file | length target | for |
|---|---|---|
| `main.tex` | **none** | the full version — journal (TGRS) or arXiv |
| `main_cvpr.tex` | **8 pages excl. references** | a conference submission (CVPR / EarthVision) |
| `supplementary.tex` | — | shared overflow, cited by both |
| `numbers.tex` | — | ⭐ **every load-bearing number, defined once** |
| `refs.bib` | — | shared bibliography |

## ⛔ Why two documents are safe here

`CLAUDE.md` recorded a decision **against** a second paper: *"one source of truth, and
`numbers.tex` is it."* The risk that decision guarded against is real — a number right in
one document and stale in the other.

**That risk is removed by construction, not by discipline.** Both papers `\input{numbers}`,
so a value changes in both or in neither. What they may differ in is *prose and emphasis*,
which is the point of having two.

`scripts/check_paper_consistency.py` enforces the rest:

1. every `\macro{}` used is defined,
2. **no result is typed inline** — a ratchet, so the known backlog in `main.tex` and
   `supplementary.tex` may fall but never rise, and `main_cvpr.tex` is held at zero,
3. environments and braces balance,
4. neither paper redefines a shared macro locally, which would silently break (1).

⚠️ **Run it before every bundle.** It has already caught two value collisions that a
careless edit would have turned into a wrong citation: `54.7` is both UAVid's published
mIoU and LoveDA `water`'s recall, and `+0.16` is both OpenEarthMap's threshold gain and
lever four's null. **Never match a macro by its value.**

## What differs between the two

`main_cvpr.tex` keeps the method, the four datasets, the CLIP transfer, the vocabulary
finding, the label-free bound and the limitations. It compresses the mechanism section
and drops to the supplementary: the co-occurrence prior, the three null levers, the
dose–response detail and the domain-transfer decomposition.

## Building

No LaTeX on the Mac side. Use `scripts/make_overleaf_bundle.sh`, which flattens
`\input{../paper/numbers}` and refuses any surviving `../`.

⚠️ **Neither document has been compiled since the UAVid and vocabulary sections were
added.** Word counts are exact; page counts are estimates until one builds.
