# Pre-registered prediction — Potsdam

- masks: **2016**  |  shapes: [(512, 512)]
- labelled pixels: **490,655,997**
- catch-all class: **`clutter`** (index 5)
- **`background` share of GT: 4.29%**

## Class composition

| class | pixels | share |
|---|---|---|
| road | 153,976,142 | 31.38% |
| building | 122,225,020 | 24.91% |
| grass | 101,226,664 | 20.63% |
| tree | 84,406,481 | 17.20% |
| clutter ⬅ catch-all | 21,036,275 | 4.29% |
| car | 7,785,415 | 1.59% |

## The anchors

| dataset | bg share | discard @ τ=0.1 | best detection AUC |
|---|---|---|---|
| LoveDA | 36.10% | 10.88% | 0.622 |
| OpenEarthMap | 0.84% | 3.78% | 0.913 |
| **Potsdam** | **4.29%** | **?** | **?** |

## The prediction — written before any inference is run

`Potsdam` sits at **4.29%**, the **covering regime**, like OpenEarthMap. Expect a small residual and detection well above the floor.

| quantity | point interpolation | honest bracket |
|---|---|---|
| real-class pixels discarded @ τ=0.1 | ~4.5% | 3.8–10.9% |
| best detection AUC | ~0.885 | 0.622–0.913 |

**Two points define a line, so no functional form is claimed.** What is falsifiable is the ORDERING against each anchor — more catch-all means more discard and worse detection:

- vs **LoveDA** (36.10% background): discard should be **LOWER than** 10.88%, and detection AUC **HIGHER than** 0.622
- vs **OpenEarthMap** (0.84% background): discard should be **HIGHER than** 3.78%, and detection AUC **LOWER than** 0.913
- detection AUC should be **well above 0.53**
- `S_pres(background)` should stay far below the real classes either way — that part is a property of SAM 3, not of the dataset, and should NOT move

> ⛔ **If a dataset near 5% background behaves like LoveDA, or one near 30% behaves like OpenEarthMap, the mechanism is wrong** and the paper's central claim needs rewriting. That is the point of running this before the pipeline rather than after.

## ⚠️ Two explanations, confounded at n=2 — which does this dataset test?

Everything so far has been attributed to the catch-all's **share**. But across the two anchors, share moves together with a second property, and the second is the more plausible cause:

| | share | **confusability** — does the catch-all LOOK like the real classes? |
|---|---|---|
| LoveDA | 36.1% | **high** — unlabelled roads, pavement and built structures that resemble `road`, `barren`, `building` |
| OpenEarthMap | 0.84% | **low** — rare genuinely-unlabelable leftovers |

`conf2` may fail on LoveDA not because background is *common*, but because background genuinely *looks like* the real classes, so a strong runner-up cannot distinguish a suppressed real class from a background region that merely resembles one. Two datasets cannot separate these.

**Decide which way this dataset breaks the tie, and write it here before running inference:**

- catch-all **visually similar** to the real classes → both hypotheses predict poor detection, and the dataset is uninformative on this question
- catch-all **visually distinct** from them → the hypotheses DISAGREE. High share predicts failure; low confusability predicts success. Either outcome eliminates one explanation.

> A dataset that can only confirm is worth less than one that can discriminate. A result this shape can be told either way afterwards, so it has to be called first.

## Next

1. Commit this file **before** running any inference.
2. `measure_discard_rate.py` with this dataset's config and its own τ.
3. `recoverability_signal.py` for the detection AUC.
4. Compare against the bracket above and record whether the prediction held.
---

## The confusability call, recorded before inference — 3 Sep 2026

Potsdam's `clutter` is defined by ISPRS as water bodies, containers, tennis courts
and similar objects: **visually distinct** from the five real classes, not a
leftover that resembles them. Combined with its **low share (4.29%)**, that means
share and confusability both predict *good* detection here.

⚠️ **So this dataset does NOT discriminate between the two explanations** — it can
only confirm the ordering, not break the tie. We record that now rather than
claiming afterwards that it settled something. The tie was already broken by
stratification within LoveDA (§7a), where share and confusability dissociate and
share wins; Potsdam is a third point on the ordering, not a second test of the
confound.

What Potsdam *does* test: whether a dataset at 4.29% behaves like the covering
regime. If its discard exceeds LoveDA's 10.88% or its detection AUC falls below
0.622, the mechanism is wrong.
