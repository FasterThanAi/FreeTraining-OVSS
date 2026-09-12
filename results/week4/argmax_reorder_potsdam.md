# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/potsdam_full/cache`  |  tiles: **2016**  |  classes: **6**, catch-all `clutter`
- published τ: **0.1**  |  5-fold  |  objective **`real`**  |  subsample **40000** px/tile

Rule: `pred = argmax_c (w_c · s_c)`, then keep `pred` if `s_pred ≥ τ_pred`. The threshold reads the **raw** score, so `w` is confined to the argmax — a scale applied afterwards is monotone and would merely reparameterise τ.

| rung | thresholds | scale | what it is |
|---|---|---|---|
| **A** | published 0.1 | 1 | the baseline |
| **B** | fitted | 1 | **our current method** |
| **C** | fitted | fitted | this experiment |

⭐ **The result is C − B.** C − A proves nothing: B already delivers that.

## Gate: does the pixel subsample reproduce the exact answer?

Rung B computed two ways — exactly over every pixel from per-tile histograms, and from the subsample the `w` search uses. The `w` result is only readable if these agree.

| fold | B−A exact | B−A subsample | difference |
|---|---|---|---|
| 1 | -0.26 | -0.21 | -0.05 |
| 2 | +0.74 | +0.74 | -0.00 |
| 3 | +0.89 | +0.88 | +0.01 |
| 4 | +0.98 | +0.97 | +0.01 |
| 5 | +0.63 | +0.68 | -0.05 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.052** mIoU against a bar of 0.15. ✅ The search subsample tracks the exact answer.

## Result

⚠️ **Rung A averages 57.84 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 58.33 | 58.07 | 62.59 | **+4.52** |
| 2 | 58.00 | 58.74 | 63.21 | **+4.47** |
| 3 | 57.35 | 58.24 | 63.40 | **+5.16** |
| 4 | 57.88 | 58.85 | 64.07 | **+5.21** |
| 5 | 57.65 | 58.28 | 63.24 | **+4.96** |

- **B − A = +0.59 ± 0.50** — the existing method, reproduced here as a sanity check.
- **C − B = +4.86 ± 0.35** — what the argmax reordering adds, 5/5 folds positive, range +4.47 to +5.21.

## Fitted scales, per fold

| fold | road | building | grass | tree | car | clutter |
|---|---|---|---|---|---|---|
| 1 | 0.64 | 0.92 | 0.64 | 4.48 | 0.44 | 1.33 |
| 2 | 0.54 | 0.96 | 0.67 | 4.05 | 0.45 | 1.61 |
| 3 | 0.43 | 0.75 | 0.61 | 3.39 | 0.43 | 3.39 |
| 4 | 0.61 | 1.05 | 0.73 | 4.02 | 0.42 | 1.26 |
| 5 | 0.63 | 0.96 | 0.68 | 4.27 | 0.44 | 1.31 |
| **mean** | **0.57** | **0.93** | **0.67** | **4.04** | **0.44** | **1.78** |
| **sd** | 0.09 | 0.11 | 0.04 | 0.41 | 0.01 | 0.91 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across folds: **51.1%** (`clutter`). ⚠️ **The fitted scales move substantially between folds**, which is what overfitting looks like. Treat the gain as unproven regardless of its sign.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| road | **+1.08** |
| building | **+0.59** |
| grass | **+3.97** |
| tree | **+21.56** |
| car | **+2.81** |
| clutter *(catch-all)* | **-0.82** |

## Verdict

⛔ **Not readable: the fitted scales move 51% between folds.** The measured increment is +4.86 ± 0.35, and its sign is not evidence either way while the parameters are this unstable — each fold is fitting a different rule.

Two causes to separate, in order. **(1) A partial or mixed cache** produces exactly this signature — compare rung A above against the dataset's published baseline first. **(2) A search subsample too thin to fit against**: `w` is searched on the subsample even though it is scored exactly, so a failed gate above and unstable scales here are the same problem, and `--subsample` is the lever.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.


## How many labelled tiles does each rule need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 3 draws each. Both rules see the SAME draw, so the columns are comparable row by row.

⚠️ This is the question a doubled parameter count forces: the ~200-tile figure in the paper was measured for thresholds alone (5 parameters). The combined rule fits 11.

| tiles | τ only | + scale | **increment** | worst increment |
|---|---|---|---|---|
| 100 | -0.09 | +4.71 | **+4.81** | +4.42 |
| 200 | +0.73 | +5.09 | **+4.36** | +4.11 |
| 400 | +0.83 | +5.52 | **+4.69** | +4.67 |
| 800 | +0.15 | +5.18 | **+5.03** | +4.41 |

⭐ Read the **increment** column, not the absolute ones. The question is the size at which the scale starts paying for its extra parameters — and whether that is larger than the figure the paper currently quotes for thresholds alone.

