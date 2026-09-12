# Per-class IoU vs its own threshold

- cache: `/home/priyanshu/outputs/potsdam/cache`  |  tiles: **2016**  |  bins: **200**  |  grid: **201** values
- calibration **200** / held out **1816**, disjoint, seed **0**  |  published τ **0.1**  |  fit objective **`real`**

Each curve sweeps one class's threshold with the other classes held at the fitted vector.

## Separability — does τ_c touch any other real class?

| swept class | max \|Δ IoU\| over other real classes | catch-all IoU range |
|---|---|---|
| road | 0.000000 | 8.95 |
| building | 0.000000 | 10.50 |
| grass | 0.000000 | 12.52 |
| tree | 0.000000 | 7.78 |
| car | 0.000000 | 2.89 |

**EXACT — the six real classes are independent given the argmax; all coupling runs through the catch-all**

One sweep costs **1005** confusion evaluations; the joint grid would cost 201^5. rounds=1 vs rounds=6 under `real`: max |Δτ| **0.0000**; under `all`: **0.0050**.

## Per class, on the held-out tiles

| class | published τ | fitted τ | calib best τ | held-out best τ | IoU @ published | @ fitted | @ held-out best |
|---|---|---|---|---|---|---|---|
| road | 0.100 | **0.085** | 0.085 | 0.055 | 73.87 | **74.55** | 74.88 |
| building | 0.100 | **0.115** | 0.115 | 0.130 | 84.96 | **84.96** | 85.03 |
| grass | 0.100 | **0.015** | 0.015 | 0.065 | 57.48 | **57.73** | 58.05 |
| tree | 0.100 | **0.055** | 0.055 | 0.035 | 37.90 | **38.23** | 38.25 |
| car | 0.100 | **0.660** | 0.660 | 0.405 | 77.03 | **80.29** | 81.21 |

⚠️ `calib best τ` and `held-out best τ` differing is the generalisation error, per class, and it is why a class can lose IoU under a fit that improved the mean (WEEK3 §9c, `road`).
