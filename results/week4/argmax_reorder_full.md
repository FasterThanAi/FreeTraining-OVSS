# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/loveda_full_all/cache`  |  tiles: **1669**  |  classes: **7**, catch-all `background`
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
| 1 | +1.25 | +1.26 | -0.01 |
| 2 | +0.97 | +0.95 | +0.02 |
| 3 | +1.13 | +1.09 | +0.04 |
| 4 | +1.35 | +1.34 | +0.01 |
| 5 | +1.17 | +1.16 | +0.01 |

Largest disagreement **0.044** mIoU against a bar of 0.15. ✅ **Gate passed** — the subsample is a valid instrument here.

## Result

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 47.17 | 48.42 | 49.38 | **+0.96** |
| 2 | 47.03 | 47.98 | 49.12 | **+1.14** |
| 3 | 47.71 | 48.80 | 50.05 | **+1.25** |
| 4 | 47.96 | 49.30 | 50.32 | **+1.01** |
| 5 | 48.29 | 49.44 | 50.89 | **+1.45** |

- **B − A = +1.16 ± 0.15** — the existing method, reproduced here as a sanity check.
- **C − B = +1.16 ± 0.19** — what the argmax reordering adds, 5/5 folds positive, range +0.96 to +1.45.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.42 | 0.59 | 1.01 | 2.54 | 1.06 | 1.46 | 1.01 |
| 2 | 0.43 | 0.57 | 0.99 | 2.67 | 1.07 | 1.47 | 0.99 |
| 3 | 0.41 | 0.60 | 0.93 | 2.48 | 1.12 | 1.50 | 1.04 |
| 4 | 0.36 | 0.60 | 0.99 | 2.59 | 1.09 | 1.58 | 1.04 |
| 5 | 0.41 | 0.58 | 0.99 | 2.47 | 1.01 | 1.71 | 1.01 |
| **mean** | **0.41** | **0.59** | **0.98** | **2.55** | **1.07** | **1.54** | **1.02** |
| **sd** | 0.03 | 0.01 | 0.03 | 0.08 | 0.04 | 0.10 | 0.02 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across folds: **6.7%** (`forest`). ⭐ **The fitted scales are stable**, so any scatter in the gains is evaluation noise from small folds rather than an unstable fit — which points at a larger cache, not at abandoning the rule.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.03** |
| building | **+0.18** |
| road | **+0.13** |
| water | **+4.41** |
| barren | **+0.63** |
| forest | **+2.63** |
| agricultural | **+0.18** |

## Verdict

⭐ **Scaling before the argmax adds +1.16 ± 0.19 mIoU over per-class thresholds alone**, every fold positive.

That is the family the completeness argument explicitly does *not* cover, so it extends the method rather than restating it. Before it is written up: (1) verify end-to-end in the segmentor, as §9c did for per-class τ — a cached-histogram result is a prediction until the pipeline reproduces it; (2) check the fitted `w` against the confusion table, since the mechanism should be visible as the classes that were absorbing other classes being scaled **down**.


⚠️ Rungs B and C are compared on **identical subsampled pixels**, so the difference is the rule and not the sample. Rung B is also reported exactly above; the exact five-fold value is **+1.17**.

