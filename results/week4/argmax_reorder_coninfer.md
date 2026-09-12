# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/coninfer_loveda_full/cache`  |  tiles: **1669**  |  classes: **7**, catch-all `background`
- published τ: **0.8**  |  5-fold  |  objective **`real`**  |  subsample **40000** px/tile

Rule: `pred = argmax_c (w_c · s_c)`, then keep `pred` if `s_pred ≥ τ_pred`. The threshold reads the **raw** score, so `w` is confined to the argmax — a scale applied afterwards is monotone and would merely reparameterise τ.

| rung | thresholds | scale | what it is |
|---|---|---|---|
| **A** | published 0.8 | 1 | the baseline |
| **B** | fitted | 1 | **our current method** |
| **C** | fitted | fitted | this experiment |

⭐ **The result is C − B.** C − A proves nothing: B already delivers that.

## Gate: does the pixel subsample reproduce the exact answer?

Rung B computed two ways — exactly over every pixel from per-tile histograms, and from the subsample the `w` search uses. The `w` result is only readable if these agree.

| fold | B−A exact | B−A subsample | difference |
|---|---|---|---|
| 1 | +2.22 | +2.20 | +0.03 |
| 2 | +2.34 | +2.33 | +0.01 |
| 3 | +3.03 | +2.76 | +0.27 |
| 4 | +2.67 | +2.68 | -0.01 |
| 5 | +2.30 | +2.27 | +0.02 |

⭐ **All three rungs below are evaluated EXACTLY, over every pixel.** The subsample is used only to search for `w`, so this gate is now a diagnostic of the search, not a condition on the result.


Largest disagreement **0.274** mIoU against a bar of 0.15. ⚠️ **The search subsample does not track the exact answer.** The results below are unaffected — they are exact — but `w` was SEARCHED on this subsample, so the fit saw a noisier objective than the one it is scored against. If the fitted scales are also unstable, raising `--subsample` is the first thing to try.

## Result

⚠️ **Rung A averages 37.00 mIoU.** That is the published baseline on these tiles — check it against the dataset's known value before reading anything below. A cache holding the wrong tiles produces a complete and plausible table.

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 37.18 | 39.40 | 39.47 | **+0.07** |
| 2 | 35.79 | 38.13 | 38.01 | **-0.12** |
| 3 | 37.97 | 41.00 | 40.72 | **-0.28** |
| 4 | 36.58 | 39.25 | 39.24 | **-0.00** |
| 5 | 37.48 | 39.78 | 39.61 | **-0.16** |

- **B − A = +2.51 ± 0.34** — the existing method, reproduced here as a sanity check.
- **C − B = -0.10 ± 0.14** — what the argmax reordering adds, 1/5 folds positive, range -0.28 to +0.07.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.60 | 0.50 | 7.07 | 0.87 | 0.60 | 1.81 |
| 2 | 0.53 | 0.53 | 0.53 | 8.36 | 0.92 | 0.53 | 1.60 |
| 3 | 0.75 | 0.49 | 0.49 | 8.32 | 0.90 | 0.49 | 1.48 |
| 4 | 0.48 | 0.58 | 0.48 | 7.51 | 0.83 | 0.58 | 2.08 |
| 5 | 0.51 | 0.51 | 0.51 | 8.43 | 0.93 | 0.51 | 1.85 |
| **mean** | **0.55** | **0.54** | **0.50** | **7.94** | **0.89** | **0.54** | **1.76** |
| **sd** | 0.11 | 0.04 | 0.02 | 0.61 | 0.04 | 0.04 | 0.23 |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

Largest relative spread across the **real** classes: **13.2%** (`agricultural`); the catch-all `background` spreads 19.8%. ⚠️ The catch-all is the least stable, which is expected: `--objective real` does not score it, so its scale is only weakly identified and its spread is not evidence about the fit. ⭐ **The fitted scales are stable**, so any scatter in the gains is evaluation noise from small folds rather than an unstable fit — which points at a larger cache, not at abandoning the rule.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **-2.31** |
| building | **-0.00** |
| road | **-0.01** |
| water | **+0.78** |
| barren | **+0.21** |
| forest | **+0.08** |
| agricultural | **+0.55** |

## Verdict

⛔ **Scaling before the argmax buys nothing: -0.10 ± 0.14 mIoU over per-class thresholds.**

A clean and useful negative. The completeness argument in the method section says per-class τ is everything available *after* the argmax; this says the obvious way to attack what lies *before* it does not pay either, at this supervision budget. State it as bounding the larger family, not as a failed attempt.


⭐ Every rung above is evaluated **exactly, over all pixels**; the subsample drove only the search for `w`.


## How many labelled tiles does each rule need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 3 draws each. Both rules see the SAME draw, so the columns are comparable row by row.

⚠️ This is the question a doubled parameter count forces: the ~200-tile figure in the paper was measured for thresholds alone (6 parameters). The combined rule fits 13.

| tiles | τ only | + scale | **increment** | worst increment |
|---|---|---|---|---|
| 25 | +1.88 | +1.70 | **-0.18** | -0.51 |
| 50 | +2.37 | +2.49 | **+0.12** | +0.06 |
| 100 | +2.37 | +2.10 | **-0.27** | -0.36 |
| 200 | +2.48 | +2.43 | **-0.05** | -0.15 |

⭐ Read the **increment** column, not the absolute ones. The question is the size at which the scale starts paying for its extra parameters — and whether that is larger than the figure the paper currently quotes for thresholds alone.

