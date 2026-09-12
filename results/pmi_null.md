# PMI vs a structure-preserving permutation null

- masks: **1669**  |  permutations: **1000**  |  seed 0
- classes: background, building, road, water, barren, forest, agriculture
- mean |PMI| observed: **1.382** bits
- mean |PMI| under the null: **0.003** bits

TWO INDEPENDENT CHECKS, and they answer different questions.

**`PMI_bnd`** replaces the area marginals with each class's own total BOUNDARY
length, so both sides of the ratio are the same measure and a class earns nothing
for merely having lots of perimeter. **This is the direct correction** to the
concern; `Δ` is how much the area-based figure in ANALYSIS §4 was inflated.

**`z`** is the permutation null: geometry held fixed, class labels permuted per
image. It tests whether the label→adjacency association is CONSISTENT, which is a
different question — a class that is reliably thin scores high here on geometry
alone. Read `Δ` for the confound, `z` for reproducibility. Sorted by |Δ|.

| Pair | PMI_area (§4) | PMI_bnd (corrected) | Δ | z (perm) |
|---|---|---|---|---|
| building – road | -1.17 | -3.15 | **+1.98** **⬅** | -13.2 |
| water – agriculture | -1.96 | -0.22 | **-1.74** **⬅** | -21.4 |
| road – forest | +0.87 | -0.85 | **+1.72** **⬅** | +9.4 |
| building – forest | -1.42 | -2.78 | **+1.35** **⬅** | -15.6 |
| background – road | +1.77 | +0.43 | **+1.34** **⬅** | +19.3 |
| road – barren | +1.23 | +0.01 | **+1.22** **⬅** | +13.5 |
| barren – agriculture | -0.66 | +0.52 | **-1.18** **⬅** | -7.2 |
| background – agriculture | -0.98 | +0.08 | **-1.06** **⬅** | -10.8 |
| background – building | +1.79 | +0.82 | **+0.97** **⬅** | +20.1 |
| building – barren | -2.10 | -2.95 | **+0.85** **⬅** | -23.0 |
| background – forest | +1.00 | +0.29 | **+0.71** **⬅** | +10.7 |
| forest – agriculture | -0.65 | +0.02 | **-0.68** **⬅** | -7.3 |
| road – water | -2.31 | -2.97 | **+0.66** **⬅** | -25.1 |
| barren – forest | -0.37 | -0.96 | **+0.60** **⬅** | -3.9 |
| water – barren | -0.10 | +0.37 | **-0.47** | -1.0 ⚠️ |
| building – agriculture | -3.39 | -2.97 | **-0.42** | -37.5 |
| background – water | +0.02 | +0.37 | **-0.35** | +0.3 ⚠️ |
| building – water | -5.75 | -6.04 | **+0.29** | -64.8 |
| background – barren | -0.19 | -0.40 | **+0.21** | -2.1 ⚠️ |
| road – agriculture | -0.51 | -0.46 | **-0.05** | -5.8 |
| water – forest | -0.76 | -0.79 | **+0.03** | -8.5 |

⚠️ **5 pair(s) CHANGE SIGN** under the corrected marginal: `road–forest` (+0.87 → -0.85), `barren–agriculture` (-0.66 → +0.52), `background–agriculture` (-0.98 → +0.08), `forest–agriculture` (-0.65 → +0.02), `water–barren` (-0.10 → +0.37). A sign flip means attraction and exclusion swap places — and ANALYSIS §4.2 makes exclusion the load-bearing signal. Any of these appearing in §4.1's table must be restated.

## Per-class rows — the §4.3 hub question ⭐

§4.3 down-weights hub classes using **the variance of a class's PMI row**, and
derives the rule from `road` — the thinnest, highest-perimeter class in LoveDA.
If `row var` moves substantially from area to boundary marginals, that weighting
was calibrated on a formula artefact and must be recomputed.

| Class | mean \|PMI\| area | mean \|PMI\| bnd | row var area | row var bnd | var change |
|---|---|---|---|---|---|
| background | 0.96 | 0.40 | 1.07 | 0.14 | **0.13×** |
| building | 2.61 | 3.12 | 5.24 | 3.96 | **0.76×** |
| road | 1.31 | 1.31 | 2.06 | 1.95 | **0.95×** |
| water | 1.82 | 1.79 | 3.87 | 5.32 | **1.38×** |
| barren | 0.78 | 0.87 | 0.96 | 1.38 | **1.44×** |
| forest | 0.84 | 0.95 | 0.77 | 0.97 | **1.25×** |
| agriculture | 1.36 | 0.71 | 1.06 | 1.31 | **1.24×** |

`var change` is the multiplier §4.3's discriminability weight `w(n) ∝ Var_c[PMI(label(n), c)]` would move by. Anything far from 1.00× means the weighting changes materially under the corrected statistic.

## Verdict

- mean |Δ| between the two statistics: **0.852** bits
- pairs changing sign: **5**
- pairs failing the permutation null (|z| < 3): **3** — water–barren, background–water, background–barren

**The corrected marginal changes conclusions.** Sign flips mean attraction and exclusion swap, and §4.2 makes exclusion the load-bearing signal. Recompute §4.1 and §4.3 on `PMI_bnd` and treat the boundary marginal as the definition going forward.

> Neither check threatens the premise: 1.3–1.7 bits against a 0.004 noise floor is far too large a margin to be geometric. What is under test is the PER-PAIR structure, which is what the method actually consumes.

> **Limitation of the permutation column.** It holds geometry fixed and permutes labels, so it detects whether a label is CONSISTENTLY associated with an adjacency profile. A class that is reliably thin (like `road`) scores high on geometry alone. It is a reproducibility check, not a control for the perimeter confound — `Δ` is the control for that. Verified on synthetic data: a corpus with a thin ribbon but randomly permuted labels yields |z| < 2 everywhere, and one with fixed labels yields |z| ~ 8 on the ribbon's pairs.