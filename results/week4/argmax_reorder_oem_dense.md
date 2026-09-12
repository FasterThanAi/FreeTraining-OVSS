# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/oem_full_v2/cache`  |  tiles: **384**  |  classes: **9**, catch-all `background`
- published τ: **0.1**  |  5-fold  |  objective **`real`**  |  subsample **150000** px/tile

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
| 1 | -1.87 | -2.16 | +0.28 |
| 2 | +0.79 | +0.40 | +0.40 |
| 3 | +1.50 | +1.60 | -0.10 |
| 4 | +1.18 | +1.33 | -0.15 |
| 5 | +0.38 | +0.23 | +0.15 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.396** mIoU against a bar of 0.15. ⚠️ **The search subsample does not track the exact answer.** The results below are unaffected — they are exact — but `w` was SEARCHED on this subsample, so the fit saw a noisier objective than the one it is scored against. If the fitted scales are also unstable, raising `--subsample` is the first thing to try.

## Result

⚠️ **Rung A averages 43.53 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 49.79 | 47.92 | 46.20 | **-1.72** |
| 2 | 39.64 | 40.43 | 40.28 | **-0.15** |
| 3 | 43.53 | 45.03 | 46.20 | **+1.17** |
| 4 | 41.65 | 42.83 | 43.57 | **+0.74** |
| 5 | 43.02 | 43.40 | 45.38 | **+1.98** |

- **B − A = +0.40 ± 1.34** — the existing method, reproduced here as a sanity check.
- **C − B = +0.40 ± 1.41** — what the argmax reordering adds, 3/5 folds positive, range -1.72 to +1.98.

## Fitted scales, per fold

| fold | background | bareland | grass | pavement | road | tree | water | cropland | building |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.56 | 0.67 | 1.16 | 1.39 | 1.44 | 1.20 | 1.45 | 1.39 | 0.48 |
| 2 | 0.42 | 0.87 | 1.14 | 1.36 | 1.43 | 1.51 | 1.05 | 1.43 | 0.55 |
| 3 | 0.53 | 1.00 | 1.10 | 1.33 | 1.33 | 1.10 | 1.59 | 1.33 | 0.42 |
| 4 | 0.53 | 0.84 | 1.21 | 1.33 | 1.33 | 1.21 | 1.60 | 1.33 | 0.41 |
| 5 | 0.86 | 0.84 | 1.03 | 1.22 | 1.24 | 1.03 | 1.49 | 1.20 | 0.48 |
| **mean** | **0.58** | **0.84** | **1.13** | **1.33** | **1.35** | **1.21** | **1.44** | **1.34** | **0.47** |
| **sd** | 0.17 | 0.12 | 0.06 | 0.07 | 0.08 | 0.18 | 0.23 | 0.08 | 0.06 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across folds: **28.6%** (`background`). ⚠️ **The fitted scales move substantially between folds**, which is what overfitting looks like. Treat the gain as unproven regardless of its sign.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.17** |
| bareland | **-4.77** |
| grass | **-1.71** |
| pavement | **+3.30** |
| road | **+0.76** |
| tree | **-0.77** |
| water | **-0.07** |
| cropland | **+7.70** |
| building | **-0.64** |

## Verdict

⛔ **Not readable: the fitted scales move 29% between folds.** The measured increment is +0.40 ± 1.41, and its sign is not evidence either way while the parameters are this unstable — each fold is fitting a different rule.

Two causes to separate, in order. **(1) A partial or mixed cache** produces exactly this signature — compare rung A above against the dataset's published baseline first. **(2) A search subsample too thin to fit against**: `w` is searched on the subsample even though it is scored exactly, so a failed gate above and unstable scales here are the same problem, and `--subsample` is the lever.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.

