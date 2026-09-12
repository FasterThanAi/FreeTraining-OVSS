# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/oem_full_v2/cache`  |  tiles: **384**  |  classes: **9**, catch-all `background`
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
| 1 | +0.45 | +0.25 | +0.20 |
| 2 | +0.05 | +0.13 | -0.08 |
| 3 | +0.54 | +0.44 | +0.10 |
| 4 | +0.27 | +0.22 | +0.05 |
| 5 | +0.75 | +0.78 | -0.03 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.199** mIoU against a bar of 0.15. ⛔ **GATE FAILED.** The subsample does not reproduce the exact answer, so nothing below can be attributed to `w` rather than to sampling noise. Raise `--subsample` and re-run.

## Result

⚠️ **Rung A averages 43.64 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 45.91 | 46.37 | 47.12 | **+0.75** |
| 2 | 46.00 | 46.05 | 45.85 | **-0.19** |
| 3 | 42.50 | 43.05 | 44.53 | **+1.48** |
| 4 | 42.74 | 43.01 | 43.71 | **+0.70** |
| 5 | 41.02 | 41.77 | 43.80 | **+2.03** |

- **B − A = +0.41 ± 0.26** — the existing method, reproduced here as a sanity check.
- **C − B = +0.95 ± 0.85** — what the argmax reordering adds, 4/5 folds positive, range -0.19 to +2.03.

## Fitted scales, per fold

| fold | background | bareland | grass | pavement | road | tree | water | cropland | building |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.47 | 0.81 | 1.17 | 1.33 | 1.40 | 1.17 | 1.68 | 1.33 | 0.47 |
| 2 | 0.53 | 0.85 | 1.10 | 1.32 | 1.32 | 1.46 | 1.47 | 1.32 | 0.41 |
| 3 | 0.57 | 0.81 | 1.04 | 1.19 | 1.25 | 1.41 | 1.69 | 1.25 | 0.47 |
| 4 | 0.69 | 0.56 | 1.16 | 1.39 | 1.44 | 1.16 | 1.20 | 1.45 | 0.56 |
| 5 | 0.52 | 0.83 | 1.08 | 1.30 | 1.30 | 1.20 | 1.56 | 1.30 | 0.52 |
| **mean** | **0.56** | **0.77** | **1.11** | **1.31** | **1.34** | **1.28** | **1.52** | **1.33** | **0.48** |
| **sd** | 0.09 | 0.12 | 0.05 | 0.07 | 0.08 | 0.15 | 0.20 | 0.07 | 0.06 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across folds: **15.7%** (`bareland`). ⚠️ **The fitted scales move substantially between folds**, which is what overfitting looks like. Treat the gain as unproven regardless of its sign.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.02** |
| bareland | **-2.66** |
| grass | **-1.77** |
| pavement | **+2.98** |
| road | **+0.78** |
| tree | **-0.80** |
| water | **+2.15** |
| cropland | **+8.57** |
| building | **-0.65** |

## Verdict

⛔ **Not readable: the fitted scales move 16% between folds.** The measured increment is +0.95 ± 0.85, and its sign is not evidence either way while the parameters are this unstable — each fold is fitting a different rule.

Check the cache before anything else: a partial or mixed cache produces exactly this signature. Compare rung A above against the dataset's published baseline; if it does not match, the input is wrong and no amount of re-reading the gain will help.


⚠️ Rungs B and C are compared on **identical subsampled pixels**, so the difference is the rule and not the sample. Rung B is also reported exactly above; the exact five-fold value is **+0.41**.


## How many labelled tiles does each rule need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 3 draws each. Both rules see the SAME draw, so the columns are comparable row by row.

⚠️ This is the question a doubled parameter count forces: the ~200-tile figure in the paper was measured for thresholds alone (8 parameters). The combined rule fits 17.

| tiles | τ only | + scale | **increment** | worst increment |
|---|---|---|---|---|
| 50 | -0.48 | +0.10 | **+0.58** | -0.16 |
| 100 | +0.22 | +0.29 | **+0.07** | -0.29 |
| 200 | +0.01 | +0.28 | **+0.27** | -0.71 |

⭐ Read the **increment** column, not the absolute ones. The question is the size at which the scale starts paying for its extra parameters — and whether that is larger than the figure the paper currently quotes for thresholds alone.

