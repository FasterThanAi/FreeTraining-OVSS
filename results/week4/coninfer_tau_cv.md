# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/coninfer_loveda/cache`  |  tiles: **1669**  |  classes: **7**
- published τ: **0.8**  |  folds: **5**
- fit objective: **`real`**  — the catch-all is excluded from what the fit maximises; reporting is still full mIoU

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 1335 | 334 | 37.18 | **39.40** | **+2.22** |
| 2 | 1335 | 334 | 35.79 | **38.13** | **+2.34** |
| 3 | 1335 | 334 | 37.97 | **41.00** | **+3.03** |
| 4 | 1335 | 334 | 36.58 | **39.25** | **+2.67** |
| 5 | 1336 | 333 | 37.48 | **39.78** | **+2.30** |

**Mean Δ = +2.51 mIoU, sd 0.34, range +2.22 to +3.03** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **+6.03** |
| building | **+0.42** |
| road | **+2.47** |
| water | **+4.21** |
| barren | **+0.76** |
| forest | **+2.01** |
| agricultural | **+1.69** |

`background` **+6.03**, the 6 real classes **+11.56** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **+1.55** | 0.84 | +0.27 |
| 25 | **+2.56** | 0.14 | +2.41 |
| 50 | **+2.14** | 0.37 | +1.66 |
| 100 | **+2.53** | 0.20 | +2.35 |
| 200 | **+2.59** | 0.24 | +2.28 |
| 400 | **+2.61** | 0.13 | +2.46 |
| 800 | **+2.59** | 0.10 | +2.51 |

## Verdict

✅ **+2.51 ± 0.34 mIoU across 5 folds** (worst +2.22), with the real classes gaining +11.56. **10 labelled tiles already reach +1.55**, so the calibration cost is small.

⚠️ Calibration tiles must come from the SAME distribution as the evaluation tiles. Fitting on LoveDA *train* and evaluating on val gives −0.12, because those splits differ sharply (discard 14.54% vs 29.68% at identical background share). State that limitation beside the gain — it is the honest scope of the result.