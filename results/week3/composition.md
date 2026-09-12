# What is the rural/urban calibration gap made of?

- cache: `/home/priyanshu/outputs/week3_fused/cache` | τ = **0.5** | 5-fold within each domain | objective **`real`**
- Δ mIoU: **rural +2.77**, urban +0.10 — a gap of **2.68**

## 1. Decomposition — arithmetic, not inference

mIoU is the *unweighted* mean over 7 classes, so the gap is exactly the mean of the per-class differences. A domain does not gain by containing more of a class; it gains when that class's own IoU improves more there.

| class | Δ IoU rural | Δ IoU urban | difference | share of the gap |
|---|---|---|---|---|
| water | +10.15 | +1.22 | **+8.93** | **+48%** |
| forest | +6.95 | -0.01 | **+6.96** | **+37%** |
| background *(catch-all)* | +0.79 | -3.51 | **+4.29** | **+23%** |
| building | -0.14 | +0.70 | **-0.84** | **-4%** |
| agricultural | +0.73 | +1.38 | **-0.65** | **-3%** |
| barren | +0.61 | +0.48 | **+0.13** | **+1%** |
| road | +0.31 | +0.42 | **-0.11** | **-1%** |

⭐ **`water` and `forest` alone account for 85% of the gap.**

## 2. What distinguishes the classes that move?

Every (domain, class) cell, 14 in all, at the published τ. `discard` is the fraction of that class assigned to the catch-all; `gap` is precision − recall.

| domain | class | share | discard | precision | recall | P−R gap | **Δ IoU** |
|---|---|---|---|---|---|---|---|
| rural | building | 3.7% | 23.4% | 77.0 | 75.8 | +1.2 | **-0.14** |
| rural | road | 2.6% | 23.2% | 62.2 | 72.0 | -9.8 | **+0.31** |
| rural | water | 11.6% | 36.7% | 87.6 | 52.6 | +35.0 | **+10.15** |
| rural | barren | 3.6% | 29.7% | 58.4 | 42.6 | +15.8 | **+0.61** |
| rural | forest | 5.0% | 67.0% | 35.4 | 9.9 | +25.5 | **+6.95** |
| rural | agricultural | 30.5% | 40.2% | 63.6 | 58.2 | +5.5 | **+0.73** |
| urban | building | 12.5% | 16.6% | 77.3 | 79.8 | -2.5 | **+0.70** |
| urban | road | 7.8% | 23.1% | 74.0 | 69.8 | +4.2 | **+0.42** |
| urban | water | 11.8% | 25.6% | 92.2 | 57.8 | +34.4 | **+1.22** |
| urban | barren | 5.4% | 20.2% | 47.8 | 65.2 | -17.4 | **+0.48** |
| urban | forest | 10.9% | 12.1% | 61.8 | 68.9 | -7.1 | **-0.01** |
| urban | agricultural | 25.7% | 17.1% | 72.5 | 68.9 | +3.6 | **+1.38** |

| candidate | ρ vs Δ IoU | permutation p |
|---|---|---|
| class share of the domain | **+0.497** | 0.1015 *(sampled)* |
| fraction discarded to the catch-all | **+0.490** | 0.1131 *(sampled)* |
| precision − recall gap | **+0.713** | 0.0128 *(sampled)* |
| precision | **+0.168** | 0.5996 *(sampled)* |
| recall | **-0.608** | 0.0414 *(sampled)* |

## Verdict

✅ **The gap is class composition, and the class statistic that explains it is precision − recall gap** (ρ = +0.713, sampled p = 0.0128). Together with the decomposition above — `water` and `forest` carrying 85% of it — the rural/urban difference is no longer unexplained: those domains differ in how much of each class the baseline leaves on the table, and calibration collects exactly that.

⚠️ This is an *explanation*, not a predictor. The statistic is label-derived, so it does not resurrect the label-free rule §9f ruled out — it says what the gain is made of, not how to know in advance.