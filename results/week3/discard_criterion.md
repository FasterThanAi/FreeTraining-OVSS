# When does per-class calibration pay? A label-free criterion

- cache: `/home/priyanshu/outputs/week3_fused/cache` | tiles: **1669** | τ = **0.5** | 4 equal-count strata | 5-fold | objective **`real`**

⭐ **The stratifying statistic uses no ground truth**: it is the fraction of all pixels the model assigns to the catch-all, computable from a forward pass over unlabelled tiles. The familiar discard rate is a different quantity — it needs labels to know which pixels held a real class. Both are shown so the substitution is visible: **Spearman ρ = +0.885** between them.

⚠️ **The random strata are the control.** Any per-tile statistic makes strata more internally homogeneous, and a homogeneous stratum may calibrate better for that reason alone. The control splits the same tiles into random strata of the **same sizes** and fits identically, so a gradient means something only if the control does not show one.

| stratum | tiles | catch-all fraction | labelled discard | **Δ mIoU** | sd | worst fold | domain mix |
|---|---|---|---|---|---|---|---|
| 1 | 418 | 0.001–0.186 | 6.0% | **+2.13** | 0.18 | +1.96 | rural 54%, urban 45% |
| 2 | 417 | 0.186–0.358 | 15.5% | **+0.78** | 0.40 | +0.16 | rural 42%, urban 57% |
| 3 | 417 | 0.359–0.767 | 27.9% | **+0.81** | 0.24 | +0.43 | rural 44%, urban 55% |
| 4 | 417 | 0.771–1.000 | 85.8% | **+3.22** | 3.15 | -1.22 | rural 96%, urban 3% |

## Control — random strata of identical sizes

| stratum | tiles | **Δ mIoU** | sd |
|---|---|---|---|
| 1 | 418 | **+1.81** | 0.74 |
| 2 | 417 | **+1.09** | 0.71 |
| 3 | 417 | **+0.62** | 0.95 |
| 4 | 417 | **+0.81** | 1.13 |

## Verdict

| | spread across strata | ρ(catch-all fraction, Δ mIoU) |
|---|---|---|
| **discard-stratified** | **2.43** | **+0.400** |
| random control | 1.19 | — |

⛔ **The gain does not track the catch-all fraction** (ρ = +0.400 over 4 strata). §9e's rural/urban difference is therefore not explained by residual size, and no label-free deployment rule follows from this statistic.

⚠️ Check the domain-mix column. If the strata are simply urban and rural re-labelled, this restates §9e rather than adding to it — the claim is only new insofar as the strata cut across the domains.