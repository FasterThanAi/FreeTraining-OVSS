# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tiles: **384**  |  classes: **9**
- published τ: **0.1**  |  folds: **5**

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 307 | 77 | 45.54 | **52.85** | **+7.31** |
| 2 | 307 | 77 | 42.31 | **49.20** | **+6.89** |
| 3 | 307 | 77 | 39.49 | **41.26** | **+1.77** |
| 4 | 307 | 77 | 44.88 | **50.77** | **+5.89** |
| 5 | 308 | 76 | 46.91 | **54.02** | **+7.11** |

**Mean Δ = +5.80 mIoU, sd 2.31, range +1.77 to +7.31** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **+53.93** |
| bareland | **-0.87** |
| grass | **+0.34** |
| pavement | **+2.71** |
| road | **-2.14** |
| tree | **-0.21** |
| water | **-0.22** |
| cropland | **-0.19** |
| building | **-1.20** |

`background` **+53.93**, the 8 real classes **-1.77** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **+3.91** | 2.14 | +0.12 |
| 25 | **+3.98** | 2.03 | +0.35 |
| 50 | **+4.35** | 1.12 | +3.13 |
| 100 | **+5.19** | 0.75 | +4.51 |
| 200 | **+5.13** | 0.27 | +4.79 |

## Verdict

⚠️ **+5.80 mIoU, but the real classes lose -1.77** — background unwinding again, not better land cover.