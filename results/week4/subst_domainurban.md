# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/loveda_full_all/cache`  |  tiles: **677**  |  classes: **7**, catch-all `background`
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
| 1 | +0.41 | +0.34 | +0.06 |
| 2 | +0.19 | +0.19 | +0.00 |
| 3 | -0.08 | -0.13 | +0.05 |
| 4 | -0.10 | +0.04 | -0.14 |
| 5 | -0.01 | +0.06 | -0.07 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.136** mIoU against a bar of 0.15. ✅ The search subsample tracks the exact answer.

## Result

⚠️ **Rung A averages 50.48 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 49.99 | 50.39 | 52.53 | **+2.14** |
| 2 | 47.93 | 48.13 | 48.72 | **+0.59** |
| 3 | 50.55 | 50.47 | 51.52 | **+1.05** |
| 4 | 51.98 | 51.88 | 53.16 | **+1.28** |
| 5 | 51.97 | 51.96 | 53.17 | **+1.21** |

- **B − A = +0.08 ± 0.21** — the existing method, reproduced here as a sanity check.
- **C − B = +1.25 ± 0.56** — what the argmax reordering adds, 5/5 folds positive, range +0.59 to +2.14.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.40 | 0.58 | 1.00 | 2.52 | 1.12 | 1.35 | 1.12 |
| 2 | 0.41 | 0.59 | 1.02 | 2.55 | 1.06 | 1.34 | 1.12 |
| 3 | 0.40 | 0.54 | 1.04 | 2.81 | 1.12 | 1.25 | 1.12 |
| 4 | 0.41 | 0.62 | 1.02 | 2.55 | 1.07 | 1.29 | 1.10 |
| 5 | 0.40 | 0.49 | 1.02 | 2.54 | 1.13 | 1.44 | 1.22 |
| **mean** | **0.40** | **0.56** | **1.02** | **2.60** | **1.10** | **1.33** | **1.14** |
| **sd** | 0.00 | 0.05 | 0.01 | 0.12 | 0.03 | 0.07 | 0.05 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across the **real** classes: **9.0%** (`building`); the catch-all `background` spreads 1.2%. ⭐ **The fitted scales are stable**, so any scatter in the gains is evaluation noise from small folds rather than an unstable fit — which points at a larger cache, not at abandoning the rule.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **+0.55** |
| building | **+0.32** |
| road | **+0.23** |
| water | **+4.71** |
| barren | **+0.10** |
| forest | **+2.19** |
| agricultural | **+0.68** |

## Verdict

⭐ **Scaling before the argmax adds +1.25 ± 0.56 mIoU over per-class thresholds alone**, every fold positive (range +0.59 to +2.14), and mean − 2·sd = +0.13.

That is the family the completeness argument explicitly does *not* cover, so it extends the method rather than restating it. Before it is written up: (1) verify end-to-end in the segmentor, as §9c did for per-class τ — a cached-histogram result is a prediction until the pipeline reproduces it; (2) read the per-class table, since a gain carried by one class in an unweighted mean is not the same claim as a broad one.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.

