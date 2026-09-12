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
| 1 | +0.16 | +0.13 | +0.03 |
| 2 | +1.25 | +1.25 | -0.00 |
| 3 | +1.74 | +1.80 | -0.06 |
| 4 | +1.03 | +0.97 | +0.07 |
| 5 | +1.37 | +1.37 | -0.00 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.068** mIoU against a bar of 0.15. ✅ The search subsample tracks the exact answer.

## Result

⚠️ **Rung A averages 47.56 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 45.92 | 46.08 | 46.86 | **+0.78** |
| 2 | 47.96 | 49.20 | 49.97 | **+0.77** |
| 3 | 48.18 | 49.91 | 50.04 | **+0.12** |
| 4 | 47.77 | 48.80 | 51.28 | **+2.47** |
| 5 | 47.97 | 49.34 | 49.67 | **+0.33** |

- **B − A = +1.11 ± 0.59** — the existing method, reproduced here as a sanity check.
- **C − B = +0.90 ± 0.93** — what the argmax reordering adds, 5/5 folds positive, range +0.12 to +2.47.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.42 | 0.67 | 0.89 | 2.43 | 1.06 | 1.50 | 1.04 |
| 2 | 0.41 | 0.65 | 0.93 | 2.33 | 1.12 | 1.49 | 1.03 |
| 3 | 0.37 | 0.63 | 0.96 | 2.39 | 1.15 | 1.58 | 1.04 |
| 4 | 0.43 | 0.69 | 0.90 | 2.50 | 1.08 | 1.26 | 1.08 |
| 5 | 0.42 | 0.67 | 0.69 | 2.60 | 1.16 | 1.67 | 1.04 |
| **mean** | **0.41** | **0.66** | **0.87** | **2.45** | **1.11** | **1.50** | **1.05** |
| **sd** | 0.03 | 0.02 | 0.11 | 0.11 | 0.04 | 0.15 | 0.02 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across the **real** classes: **12.1%** (`road`); the catch-all `background` spreads 6.2%. ⭐ **The fitted scales are stable**, so any scatter in the gains is evaluation noise from small folds rather than an unstable fit — which points at a larger cache, not at abandoning the rule.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.24** |
| building | **+0.17** |
| road | **-0.06** |
| water | **+2.80** |
| barren | **+0.26** |
| forest | **+3.27** |
| agricultural | **+0.07** |

## Verdict

⚠️ **Promising, not established: +0.90 ± 0.93 mIoU over per-class thresholds, every fold positive** (range +0.12 to +2.47).

Every fold positive is real evidence — under a sign test that is 1 in 32. But the spread covers zero (mean − 2·sd = -0.96), which is the bar this project uses elsewhere, so this does not yet clear it. **Do not write it up as a result at this width.** Two things close it, in order: a larger cache so the folds stop being small, and an end-to-end run in the segmentor as §9c did for per-class τ.

⭐ Check the fitted-scale stability table above before anything else. If the scales agree across folds while the gains scatter, the *fit* is stable and the *evaluation* is noisy — which argues for more evaluation tiles rather than against the effect.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.

