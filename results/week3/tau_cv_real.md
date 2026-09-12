# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tiles: **1669**  |  classes: **7**
- published τ: **0.5**  |  folds: **5**
- fit objective: **`real`**  — the catch-all is excluded from what the fit maximises; reporting is still full mIoU

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 1335 | 334 | 47.91 | **49.28** | **+1.37** |
| 2 | 1335 | 334 | 45.41 | **47.31** | **+1.89** |
| 3 | 1335 | 334 | 48.44 | **49.32** | **+0.88** |
| 4 | 1335 | 334 | 46.11 | **46.91** | **+0.80** |
| 5 | 1336 | 333 | 48.68 | **49.66** | **+0.98** |

**Mean Δ = +1.18 mIoU, sd 0.45, range +0.80 to +1.89** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **-0.01** |
| building | **+0.28** |
| road | **+0.10** |
| water | **+6.78** |
| barren | **+0.98** |
| forest | **+0.37** |
| agricultural | **-0.21** |

`background` **-0.01**, the 6 real classes **+8.30** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **-2.14** | 1.99 | -5.59 |
| 25 | **-1.51** | 1.75 | -3.74 |
| 50 | **+0.08** | 0.97 | -0.99 |
| 100 | **+0.54** | 0.71 | -0.57 |
| 200 | **+0.79** | 0.35 | +0.43 |
| 400 | **+1.21** | 0.16 | +1.04 |
| 800 | **+0.93** | 0.41 | +0.20 |

## Verdict

✅ **+1.18 ± 0.45 mIoU across 5 folds** (worst +0.80), with the real classes gaining +8.30. **200 labelled tiles already reach +0.79**, so the calibration cost is small.

⚠️ Calibration tiles must come from the SAME distribution as the evaluation tiles. Fitting on LoveDA *train* and evaluating on val gives −0.12, because those splits differ sharply (discard 14.54% vs 29.68% at identical background share). State that limitation beside the gain — it is the honest scope of the result.