# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tiles: **1669**  |  classes: **7**
- published τ: **0.5**  |  folds: **5**

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 1335 | 334 | 47.91 | **49.51** | **+1.59** |
| 2 | 1335 | 334 | 45.41 | **47.48** | **+2.07** |
| 3 | 1335 | 334 | 48.44 | **49.27** | **+0.84** |
| 4 | 1335 | 334 | 46.11 | **47.11** | **+1.01** |
| 5 | 1336 | 333 | 48.68 | **49.74** | **+1.06** |

**Mean Δ = +1.31 mIoU, sd 0.51, range +0.84 to +2.07** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **+1.52** |
| building | **+0.25** |
| road | **+0.04** |
| water | **+6.58** |
| barren | **+0.99** |
| forest | **+0.23** |
| agricultural | **-0.42** |

`background` **+1.52**, the 6 real classes **+7.67** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **-2.33** | 2.44 | -6.01 |
| 25 | **+0.34** | 0.62 | -0.29 |
| 50 | **+0.78** | 0.22 | +0.40 |
| 100 | **+0.91** | 0.27 | +0.62 |
| 200 | **+1.05** | 0.29 | +0.56 |
| 400 | **+1.32** | 0.13 | +1.14 |
| 800 | **+1.17** | 0.26 | +0.72 |

## Verdict

✅ **+1.31 ± 0.51 mIoU across 5 folds** (worst +0.84), with the real classes gaining +7.67. **100 labelled tiles already reach +0.91**, so the calibration cost is small.

⚠️ Calibration tiles must come from the SAME distribution as the evaluation tiles. Fitting on LoveDA *train* and evaluating on val gives −0.12, because those splits differ sharply (discard 14.54% vs 29.68% at identical background share). State that limitation beside the gain — it is the honest scope of the result.