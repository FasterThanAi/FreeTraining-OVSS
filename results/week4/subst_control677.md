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
| 1 | +1.25 | +1.22 | +0.03 |
| 2 | -0.15 | -0.07 | -0.08 |
| 3 | +0.12 | +0.05 | +0.07 |
| 4 | +2.65 | +2.51 | +0.14 |
| 5 | +1.22 | +1.20 | +0.02 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.140** mIoU against a bar of 0.15. ✅ The search subsample tracks the exact answer.

## Result

⚠️ **Rung A averages 47.46 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 48.39 | 49.64 | 49.49 | **-0.15** |
| 2 | 49.00 | 48.84 | 51.55 | **+2.71** |
| 3 | 49.07 | 49.18 | 49.03 | **-0.15** |
| 4 | 43.80 | 46.44 | 47.23 | **+0.79** |
| 5 | 47.03 | 48.26 | 48.34 | **+0.08** |

- **B − A = +1.02 ± 1.11** — the existing method, reproduced here as a sanity check.
- **C − B = +0.65 ± 1.21** — what the argmax reordering adds, 3/5 folds positive, range -0.15 to +2.71.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.70 | 0.58 | 0.84 | 2.24 | 0.89 | 1.75 | 0.84 |
| 2 | 0.40 | 0.69 | 1.00 | 2.50 | 1.00 | 1.44 | 1.00 |
| 3 | 0.41 | 0.81 | 0.98 | 2.52 | 1.01 | 1.24 | 0.98 |
| 4 | 0.42 | 0.58 | 1.00 | 2.65 | 1.01 | 1.74 | 0.88 |
| 5 | 0.76 | 0.54 | 0.84 | 2.36 | 0.92 | 1.32 | 1.00 |
| **mean** | **0.54** | **0.64** | **0.93** | **2.45** | **0.96** | **1.50** | **0.94** |
| **sd** | 0.18 | 0.11 | 0.09 | 0.16 | 0.05 | 0.23 | 0.07 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across the **real** classes: **17.3%** (`building`); the catch-all `background` spreads 32.8%. ⚠️ The catch-all is the least stable, which is expected: `--objective real` does not score it, so its scale is only weakly identified and its spread is not evidence about the fit. ⚠️ **The fitted scales move substantially between folds**, which is what overfitting looks like. Treat the gain as unproven regardless of its sign.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.78** |
| building | **+0.08** |
| road | **+0.08** |
| water | **+2.80** |
| barren | **-0.59** |
| forest | **+3.35** |
| agricultural | **-0.36** |

## Verdict

⛔ **Not readable: the fitted scales move 17% between folds.** The measured increment is +0.65 ± 1.21, and its sign is not evidence either way while the parameters are this unstable — each fold is fitting a different rule.

Two causes to separate, in order. **(1) A partial or mixed cache** produces exactly this signature — compare rung A above against the dataset's published baseline first. **(2) A search subsample too thin to fit against**: `w` is searched on the subsample even though it is scored exactly, so a failed gate above and unstable scales here are the same problem, and `--subsample` is the lever.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.

