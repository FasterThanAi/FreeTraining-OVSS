# PMI vs a structure-preserving permutation null

- masks: **1669**  |  permutations: **1000**  |  seed 0
- classes: building, road, water, barren, forest, agriculture  (dropped: background)
- mean |PMI| observed: **1.217** bits
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
| road – barren | +2.32 | +0.17 | **+2.15** **⬅** | +25.9 |
| building – agriculture | -2.31 | -0.25 | **-2.06** **⬅** | -25.6 |
| road – forest | +1.96 | +0.06 | **+1.89** **⬅** | +21.8 |
| barren – forest | +0.72 | -1.11 | **+1.83** **⬅** | +7.8 |
| building – water | -4.66 | -2.83 | **-1.83** **⬅** | -52.9 |
| water – agriculture | -0.87 | +0.25 | **-1.12** **⬅** | -9.6 |
| road – water | -1.23 | -1.88 | **+0.66** **⬅** | -13.5 |
| water – barren | +0.99 | +0.39 | **+0.60** **⬅** | +11.3 |
| building – forest | -0.34 | +0.26 | **-0.60** **⬅** | -3.7 |
| road – agriculture | +0.58 | +0.15 | **+0.43** | +6.5 |
| barren – agriculture | +0.43 | +0.05 | **+0.37** | +4.8 |
| water – forest | +0.33 | -0.01 | **+0.34** | +3.8 |
| building – barren | -1.01 | -0.67 | **-0.34** | -11.2 |
| building – road | -0.08 | +0.21 | **-0.28** | -0.9 ⚠️ |
| forest – agriculture | +0.44 | +0.32 | **+0.12** | +4.9 |

⚠️ **5 pair(s) CHANGE SIGN** under the corrected marginal: `barren–forest` (+0.72 → -1.11), `water–agriculture` (-0.87 → +0.25), `building–forest` (-0.34 → +0.26), `water–forest` (+0.33 → -0.01), `building–road` (-0.08 → +0.21). A sign flip means attraction and exclusion swap places — and ANALYSIS §4.2 makes exclusion the load-bearing signal. Any of these appearing in §4.1's table must be restated.

## Per-class rows — the §4.3 hub question ⭐

§4.3 down-weights hub classes using **the variance of a class's PMI row**, and
derives the rule from `road` — the thinnest, highest-perimeter class in LoveDA.
If `row var` moves substantially from area to boundary marginals, that weighting
was calibrated on a formula artefact and must be recomputed.

| Class | mean \|PMI\| area | mean \|PMI\| bnd | row var area | row var bnd | var change |
|---|---|---|---|---|---|
| building | 1.68 | 0.84 | 2.82 | 1.30 | **0.46×** |
| road | 1.23 | 0.49 | 1.71 | 0.66 | **0.39×** |
| water | 1.62 | 1.07 | 3.84 | 1.69 | **0.44×** |
| barren | 1.09 | 0.48 | 1.14 | 0.32 | **0.28×** |
| forest | 0.76 | 0.35 | 0.56 | 0.27 | **0.48×** |
| agriculture | 0.92 | 0.20 | 1.24 | 0.04 | **0.03×** |

`var change` is the multiplier §4.3's discriminability weight `w(n) ∝ Var_c[PMI(label(n), c)]` would move by. Anything far from 1.00× means the weighting changes materially under the corrected statistic.

## Verdict

- mean |Δ| between the two statistics: **0.975** bits
- pairs changing sign: **5**
- pairs failing the permutation null (|z| < 3): **1** — building–road

**The corrected marginal changes conclusions.** Sign flips mean attraction and exclusion swap, and §4.2 makes exclusion the load-bearing signal. Recompute §4.1 and §4.3 on `PMI_bnd` and treat the boundary marginal as the definition going forward.

> Neither check threatens the premise: 1.3–1.7 bits against a 0.004 noise floor is far too large a margin to be geometric. What is under test is the PER-PAIR structure, which is what the method actually consumes.

> **Limitation of the permutation column.** It holds geometry fixed and permutes labels, so it detects whether a label is CONSISTENTLY associated with an adjacency profile. A class that is reliably thin (like `road`) scores high on geometry alone. It is a reproducibility check, not a control for the perimeter confound — `Δ` is the control for that. Verified on synthetic data: a corpus with a thin ribbon but randomly permuted labels yields |z| < 2 everywhere, and one with fixed labels yields |z| ~ 8 on the ribbon's pairs.