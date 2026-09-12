# When does per-class calibration pay? A label-free criterion

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache` | tiles: **384** | τ = **0.1** | 3 equal-count strata | 5-fold | objective **`real`**

⭐ **The stratifying statistic uses no ground truth**: it is the fraction of all pixels the model assigns to the catch-all, computable from a forward pass over unlabelled tiles. The familiar discard rate is a different quantity — it needs labels to know which pixels held a real class. Both are shown so the substitution is visible: **Spearman ρ = +0.924** between them.

⚠️ **The random strata are the control.** Any per-tile statistic makes strata more internally homogeneous, and a homogeneous stratum may calibrate better for that reason alone. The control splits the same tiles into random strata of the **same sizes** and fits identically, so a gradient means something only if the control does not show one.

| stratum | tiles | catch-all fraction | labelled discard | **Δ mIoU** | sd | worst fold |
|---|---|---|---|---|---|---|
| 1 | 128 | 0.000–0.011 | 0.3% | **+0.51** | 1.22 | -0.79 |
| 2 | 128 | 0.012–0.046 | 2.7% | **-0.08** | 2.23 | -3.94 |
| 3 | 128 | 0.046–0.410 | 8.4% | **+0.47** | 0.58 | -0.11 |

## Control — random strata of identical sizes

| stratum | tiles | **Δ mIoU** | sd |
|---|---|---|---|
| 1 | 128 | **-0.10** | 1.69 |
| 2 | 128 | **+0.06** | 1.73 |
| 3 | 128 | **+0.32** | 1.03 |

## Verdict

| | spread across strata | ρ(catch-all fraction, Δ mIoU) |
|---|---|---|
| **discard-stratified** | **0.59** | **-0.500** |
| random control | 0.41 | — |

⚠️ **Inconclusive — the control moves nearly as much.** Discard strata span 0.59 against 0.41 for random strata of the same sizes. Splitting the data at all produces most of this variation, so the catch-all fraction is not shown to be what matters. Report the per-stratum table and no rule.