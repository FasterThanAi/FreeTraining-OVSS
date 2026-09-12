# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tiles: **384**  |  classes: **9**
- published τ: **0.1**  |  folds: **5**
- fit objective: **`real`**  — the catch-all is excluded from what the fit maximises; reporting is still full mIoU

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 307 | 77 | 45.54 | **44.27** | **-1.27** |
| 2 | 307 | 77 | 42.31 | **42.98** | **+0.67** |
| 3 | 307 | 77 | 39.49 | **40.63** | **+1.14** |
| 4 | 307 | 77 | 44.88 | **44.71** | **-0.18** |
| 5 | 308 | 76 | 46.91 | **47.34** | **+0.43** |

**Mean Δ = +0.16 mIoU, sd 0.93, range -1.27 to +1.14** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **-11.04** |
| bareland | **+0.29** |
| grass | **+0.34** |
| pavement | **+2.71** |
| road | **+2.95** |
| tree | **+0.01** |
| water | **-0.24** |
| cropland | **+2.04** |
| building | **+4.35** |

`background` **-11.04**, the 8 real classes **+12.45** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **-0.60** | 1.52 | -3.31 |
| 25 | **-0.69** | 1.08 | -2.02 |
| 50 | **-0.39** | 1.15 | -2.40 |
| 100 | **+0.37** | 0.24 | +0.01 |
| 200 | **+0.12** | 0.79 | -1.24 |

## Verdict

⛔ **Not distinguishable from zero.** Mean +0.16 with sd 0.93 over 5 folds, so the spread covers no gain at all. The single-split +1.44 was a favourable draw. Report the oracle bound only.