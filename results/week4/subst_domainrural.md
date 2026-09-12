# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/loveda_full_all/cache`  |  tiles: **992**  |  classes: **7**, catch-all `background`
- published τ: **0.5**  |  5-fold  |  objective **`real`**  |  subsample **40000** px/tile

Rule: `pred = argmax_c (w_c · s_c)`, then keep `pred` if `s_pred ≥ τ_pred`. The threshold reads the **raw** score, so `w` is confined to the argmax — a scale applied afterwards is monotone and would merely reparameterise τ.

| rung | thresholds | scale | what it is |
|---|---|---|---|
| **A** | published 0.5 | 1 | the baseline |
| **B** | fitted | 1 | **our current method** |
| **C** | fitted | fitted | this experiment |

⭐ **The result is C − B.** C − A proves nothing: B already delivers that.

## Gate: does the pixel subsample reproduce the exact answer?

Rung B computed two ways — exactly over every pixel from per-tile histograms, and from the subsample the `w` search uses. The `w` result is only readable if these agree.

| fold | B−A exact | B−A subsample | difference |
|---|---|---|---|
| 1 | +3.49 | +3.51 | -0.01 |
| 2 | +3.74 | +3.66 | +0.08 |
| 3 | +1.61 | +1.61 | +0.00 |
| 4 | +2.99 | +3.01 | -0.03 |
| 5 | +2.03 | +2.01 | +0.02 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.081** mIoU against a bar of 0.15. ✅ The search subsample tracks the exact answer.

## Result

⚠️ **Rung A averages 42.00 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 41.33 | 44.82 | 46.62 | **+1.80** |
| 2 | 41.31 | 45.04 | 45.42 | **+0.38** |
| 3 | 42.57 | 44.18 | 45.68 | **+1.50** |
| 4 | 40.66 | 43.65 | 44.94 | **+1.29** |
| 5 | 44.16 | 46.18 | 47.32 | **+1.14** |

- **B − A = +2.77 ± 0.92** — the existing method, reproduced here as a sanity check.
- **C − B = +1.22 ± 0.53** — what the argmax reordering adds, 5/5 folds positive, range +0.38 to +1.80.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.38 | 0.56 | 0.68 | 2.47 | 1.14 | 2.47 | 0.99 |
| 2 | 0.38 | 0.65 | 0.65 | 2.50 | 1.13 | 2.35 | 0.94 |
| 3 | 0.37 | 0.69 | 0.63 | 2.49 | 1.10 | 2.49 | 0.92 |
| 4 | 0.38 | 0.69 | 0.55 | 2.53 | 1.15 | 2.40 | 0.96 |
| 5 | 0.30 | 0.55 | 1.62 | 2.40 | 0.89 | 2.40 | 0.74 |
| **mean** | **0.36** | **0.63** | **0.83** | **2.48** | **1.08** | **2.43** | **0.91** |
| **sd** | 0.04 | 0.07 | 0.44 | 0.05 | 0.11 | 0.06 | 0.10 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across the **real** classes: **53.7%** (`road`); the catch-all `background` spreads 10.3%. ⚠️ **The fitted scales move substantially between folds**, which is what overfitting looks like. Treat the gain as unproven regardless of its sign.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-3.20** |
| building | **+0.14** |
| road | **+0.10** |
| water | **+4.34** |
| barren | **+1.41** |
| forest | **+4.98** |
| agricultural | **+0.78** |

## Verdict

⭐ **Scaling before the argmax adds +1.22 ± 0.53 mIoU over per-class thresholds alone**, every fold positive (range +0.38 to +1.80), and mean − 2·sd = +0.16.

That is the family the completeness argument explicitly does *not* cover, so it extends the method rather than restating it. Before it is written up: (1) verify end-to-end in the segmentor, as §9c did for per-class τ — a cached-histogram result is a prediction until the pipeline reproduces it; (2) read the per-class table, since a gain carried by one class in an unweighted mean is not the same claim as a broad one.

⚠️ The fitted scales for the real classes spread up to 54% across folds while the increment holds to ±0.53. That is **non-uniqueness, not overfitting** — different scale vectors reaching equivalent optima on a coupled objective. Report the gain; do not present any single fitted vector as *the* answer.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.

