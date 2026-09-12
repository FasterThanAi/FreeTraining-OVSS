# Per-class IoU vs its own threshold

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tiles: **384**  |  bins: **200**  |  grid: **201** values
- calibration **100** / held out **284**, disjoint, seed **0**  |  published τ **0.1**  |  fit objective **`real`**

Each curve sweeps one class's threshold with the other classes held at the fitted vector.

## Separability — does τ_c touch any other real class?

| swept class | max \|Δ IoU\| over other real classes | catch-all IoU range |
|---|---|---|
| bareland | 0.000000 | 1.74 |
| grass | 0.000000 | 2.69 |
| pavement | 0.000000 | 1.91 |
| road | 0.000000 | 2.09 |
| tree | 0.000000 | 2.78 |
| water | 0.000000 | 0.60 |
| cropland | 0.000000 | 3.28 |
| building | 0.000000 | 4.03 |

**EXACT — the six real classes are independent given the argmax; all coupling runs through the catch-all**

One sweep costs **1608** confusion evaluations; the joint grid would cost 201^8. rounds=1 vs rounds=6 under `real`: max |Δτ| **0.0000**; under `all`: **0.0000**.

## Per class, on the held-out tiles

| class | published τ | fitted τ | calib best τ | held-out best τ | IoU @ published | @ fitted | @ held-out best |
|---|---|---|---|---|---|---|---|
| bareland | 0.100 | **0.185** | 0.185 | 0.250 | 11.53 | **14.67** | 16.57 |
| grass | 0.100 | **0.050** | 0.050 | 0.040 | 43.30 | **43.64** | 43.65 |
| pavement | 0.100 | **0.015** | 0.015 | 0.020 | 27.65 | **30.26** | 30.26 |
| road | 0.100 | **0.310** | 0.310 | 0.550 | 46.05 | **49.09** | 49.64 |
| tree | 0.100 | **0.220** | 0.220 | 0.140 | 62.99 | **62.76** | 63.16 |
| water | 0.100 | **0.710** | 0.710 | 0.240 | 69.18 | **55.02** | 69.84 |
| cropland | 0.100 | **0.550** | 0.550 | 0.520 | 42.12 | **45.41** | 45.59 |
| building | 0.100 | **0.375** | 0.375 | 0.385 | 75.44 | **79.66** | 79.66 |

⚠️ `calib best τ` and `held-out best τ` differing is the generalisation error, per class, and it is why a class can lose IoU under a fit that improved the mean (WEEK3 §9c, `road`).
