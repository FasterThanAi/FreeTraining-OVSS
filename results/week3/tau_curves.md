# Per-class IoU vs its own threshold

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tiles: **1669**  |  bins: **200**  |  grid: **201** values
- calibration **200** / held out **1469**, disjoint, seed **0**  |  published τ **0.5**  |  fit objective **`real`**

Each curve sweeps one class's threshold with the other classes held at the fitted vector.

## Separability — does τ_c touch any other real class?

| swept class | max \|Δ IoU\| over other real classes | catch-all IoU range |
|---|---|---|
| building | 0.000000 | 2.03 |
| road | 0.000000 | 3.15 |
| water | 0.000000 | 4.17 |
| barren | 0.000000 | 9.08 |
| forest | 0.000000 | 3.11 |
| agricultural | 0.000000 | 17.38 |

**EXACT — the six real classes are independent given the argmax; all coupling runs through the catch-all**

One sweep costs **1206** confusion evaluations; the joint grid would cost 201^6. rounds=1 vs rounds=6 under `real`: max |Δτ| **0.0000**; under `all`: **0.0000**.

## Per class, on the held-out tiles

| class | published τ | fitted τ | calib best τ | held-out best τ | IoU @ published | @ fitted | @ held-out best |
|---|---|---|---|---|---|---|---|
| building | 0.500 | **0.190** | 0.190 | 0.430 | 64.12 | **64.25** | 64.82 |
| road | 0.500 | **0.675** | 0.675 | 0.540 | 53.98 | **53.44** | 54.07 |
| water | 0.500 | **0.175** | 0.175 | 0.160 | 50.88 | **57.42** | 57.45 |
| barren | 0.500 | **0.375** | 0.375 | 0.375 | 35.20 | **36.27** | 36.27 |
| forest | 0.500 | **0.410** | 0.410 | 0.400 | 33.42 | **33.86** | 33.93 |
| agricultural | 0.500 | **0.565** | 0.565 | 0.505 | 47.18 | **46.93** | 47.20 |

⚠️ `calib best τ` and `held-out best τ` differing is the generalisation error, per class, and it is why a class can lose IoU under a fit that improved the mean (WEEK3 §9c, `road`).

reference check vs WEEK3 §9c: ✅ every threshold identical
