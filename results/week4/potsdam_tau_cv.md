# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/potsdam/cache`  |  tiles: **2016**  |  classes: **6**
- published τ: **0.1**  |  folds: **5**
- fit objective: **`real`**  — the catch-all is excluded from what the fit maximises; reporting is still full mIoU

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 1612 | 404 | 58.33 | **58.07** | **-0.26** |
| 2 | 1613 | 403 | 58.00 | **58.74** | **+0.74** |
| 3 | 1613 | 403 | 57.35 | **58.24** | **+0.89** |
| 4 | 1613 | 403 | 57.88 | **58.86** | **+0.98** |
| 5 | 1613 | 403 | 57.66 | **58.28** | **+0.63** |

**Mean Δ = +0.59 mIoU, sd 0.50, range -0.26 to +0.98** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| road | **+0.70** |
| building | **-0.17** |
| grass | **+0.68** |
| tree | **+0.32** |
| car | **+3.69** |
| clutter *(catch-all)* | **-1.65** |

`clutter` **-1.65**, the 5 real classes **+5.22** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **-1.99** | 3.23 | -7.74 |
| 25 | **+0.17** | 0.48 | -0.61 |
| 50 | **+0.34** | 0.37 | -0.11 |
| 100 | **+0.26** | 0.23 | -0.02 |
| 200 | **+0.60** | 0.30 | +0.10 |
| 400 | **+0.47** | 0.20 | +0.28 |
| 800 | **+0.43** | 0.36 | +0.20 |

## Verdict

⛔ **Not distinguishable from zero.** Mean +0.59 with sd 0.50 over 5 folds, so the spread covers no gain at all. The single-split +1.44 was a favourable draw. Report the oracle bound only.