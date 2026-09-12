# Per-class τ, fitted on train and evaluated on val

- train: **2522** tiles (`/home/priyanshu/outputs/loveda_train/cache`)
- val: **1669** tiles (`/home/priyanshu/outputs/week3_fused/cache`)
- classes: **7**  |  published τ: **0.5**  |  no tile id in both

SegEarth-OV3 tunes τ per dataset using labels, so fitting thresholds on a train split and evaluating on val is **the same protocol with more parameters**. No model weights are trained; the pipeline stays training-free.

| rule | params | fitted on | train mIoU | **val mIoU** | Δ val |
|---|---|---|---|---|---|
| published τ = 0.5 | 1 | (theirs) | 49.56 | **47.37** | — |
| global τ = 0.530 | 1 | train | 49.64 | **47.19** | -0.18 |
| **per-class τ** | 6 | **train** | 50.48 | **47.25** | **-0.12** |
| _per-class τ (oracle)_ | 6 | _val_ | — | _48.83_ | _+1.46_ |

Generalisation gap on the fitted row: train 50.48 → val 47.25 (**-3.23**). The oracle row is fitted on val and is a ceiling, never a result.

## Thresholds

| class | fitted on train | oracle (val) | published |
|---|---|---|---|
| background | — *(no effect)* | — | — |
| building | **0.205** | 0.320 | 0.5 |
| road | **0.575** | 0.575 | 0.5 |
| water | **0.500** | 0.170 | 0.5 |
| barren | **0.380** | 0.375 | 0.5 |
| forest | **0.525** | 0.440 | 0.5 |
| agricultural | **0.685** | 0.595 | 0.5 |

## Per-class IoU on val, published τ vs fitted

| class | published | fitted | Δ |
|---|---|---|---|
| background | 45.51 | 47.23 | **+1.72** |
| building | 63.80 | 64.40 | **+0.60** |
| road | 53.88 | 53.99 | **+0.11** |
| water | 51.41 | 51.41 | **+0.00** |
| barren | 35.75 | 36.75 | **+1.00** |
| forest | 33.75 | 33.41 | **-0.33** |
| agricultural | 47.48 | 43.56 | **-3.93** |

`background` **+1.72**, the 6 real classes **-2.56** in aggregate.

## Verdict

⛔ **The fitted thresholds do not transfer** (-0.12 on val against +0.92 on train). The oracle gain is real but only reachable by fitting on the evaluation set, so per-class thresholding is a property of the split rather than a method. Report the bound, not a method.