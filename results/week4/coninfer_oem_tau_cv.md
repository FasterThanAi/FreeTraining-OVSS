# Per-class τ — cross-validated

- cache: `/home/priyanshu/outputs/coninfer_oem/cache`  |  tiles: **384**  |  classes: **9**
- published τ: **0.1**  |  folds: **5**
- fit objective: **`real`**  — the catch-all is excluded from what the fit maximises; reporting is still full mIoU

Calibration and evaluation tiles are always disjoint, and the published-τ baseline is recomputed on the same held-out tiles, so both are measured on identical pixels.

| fold | calib tiles | eval tiles | published τ | fitted | Δ |
|---|---|---|---|---|---|
| 1 | 307 | 77 | 29.88 | **31.70** | **+1.82** |
| 2 | 307 | 77 | 26.65 | **25.44** | **-1.22** |
| 3 | 307 | 77 | 28.94 | **28.19** | **-0.74** |
| 4 | 307 | 77 | 31.32 | **29.99** | **-1.32** |
| 5 | 308 | 76 | 30.75 | **30.24** | **-0.50** |

**Mean Δ = -0.39 mIoU, sd 1.28, range -1.32 to +1.82** over 5 folds.

## Mean per-class Δ IoU across folds

| class | Δ |
|---|---|
| background *(catch-all)* | **-11.16** |
| bareland | **+5.88** |
| grass | **-0.00** |
| pavement | **+0.00** |
| road | **+0.02** |
| tree | **+0.01** |
| water | **+1.42** |
| cropland | **+0.26** |
| building | **+0.04** |

`background` **-11.16**, the 8 real classes **+7.62** in aggregate.

## How many labelled tiles does calibration need?

Fit on *n* randomly drawn tiles, evaluate on the rest, 5 draws each.

| calib tiles | mean Δ | sd | worst draw |
|---|---|---|---|
| 10 | **-1.07** | 1.28 | -2.49 |
| 25 | **-1.00** | 0.41 | -1.55 |
| 50 | **-0.51** | 0.33 | -1.03 |
| 100 | **-0.57** | 0.55 | -1.40 |
| 200 | **-0.33** | 0.31 | -0.68 |

## Verdict

⛔ **Not distinguishable from zero.** Mean -0.39 with sd 1.28 over 5 folds, so the spread covers no gain at all. The single-split +1.44 was a favourable draw. Report the oracle bound only.