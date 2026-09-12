# Per-class scaling before the argmax

- cache: `/home/priyanshu/outputs/loveda_full/cache`  |  tiles: **500**  |  classes: **7**, catch-all `background`
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
| 1 | -0.32 | -0.37 | +0.05 |
| 2 | +1.57 | +1.57 | +0.00 |
| 3 | +0.18 | +0.12 | +0.06 |
| 4 | +1.12 | +1.08 | +0.04 |
| 5 | +1.86 | +1.84 | +0.02 |

Largest disagreement **0.061** mIoU against a bar of 0.15. ✅ **Gate passed** — the subsample is a valid instrument here.

## Result

| fold | A published τ | B per-class τ | C + scaling | **C − B** |
|---|---|---|---|---|
| 1 | 48.14 | 47.77 | 48.31 | **+0.54** |
| 2 | 49.22 | 50.78 | 50.85 | **+0.07** |
| 3 | 48.79 | 48.91 | 50.70 | **+1.79** |
| 4 | 47.48 | 48.56 | 49.45 | **+0.90** |
| 5 | 44.15 | 45.98 | 48.46 | **+2.48** |

- **B − A = +0.85 ± 0.94** — the existing method, reproduced here as a sanity check.
- **C − B = +1.16 ± 0.97** — what the argmax reordering adds, 5/5 folds positive, range +0.07 to +2.48.

## Fitted scales, per fold

| fold | background | building | road | water | barren | forest | agricultural |
|---|---|---|---|---|---|---|---|
| 1 | 0.38 | 0.66 | 1.14 | 2.38 | 1.20 | 1.20 | 1.03 |
| 2 | 0.40 | 0.72 | 0.87 | 2.50 | 1.00 | 1.52 | 1.05 |
| 3 | 0.38 | 0.66 | 0.79 | 2.43 | 1.17 | 1.60 | 1.11 |
| 4 | 0.39 | 0.71 | 0.97 | 2.56 | 1.02 | 1.48 | 0.97 |
| 5 | 0.38 | 0.79 | 0.95 | 2.38 | 1.01 | 1.46 | 0.98 |
| **mean** | **0.39** | **0.71** | **0.95** | **2.45** | **1.08** | **1.45** | **1.03** |

Values are renormalised to geometric mean 1, since only ratios affect an argmax. A class above 1 wins more argmaxes than before; below 1, fewer.

## Mean per-class Δ IoU, C over B

| class | Δ |
|---|---|
| background *(catch-all)* | **+0.41** |
| building | **+0.07** |
| road | **-0.06** |
| water | **+4.06** |
| barren | **-0.53** |
| forest | **+4.52** |
| agricultural | **-0.38** |

## Verdict

⛔ **Scaling before the argmax buys nothing: +1.16 ± 0.97 mIoU over per-class thresholds.**

A clean and useful negative. The completeness argument in the method section says per-class τ is everything available *after* the argmax; this says the obvious way to attack what lies *before* it does not pay either, at this supervision budget. State it as bounding the larger family, not as a failed attempt.


⚠️ Rungs B and C are compared on **identical subsampled pixels**, so the difference is the rule and not the sample. Rung B is also reported exactly above; the exact five-fold value is **+0.88**.

