# Per-class τ, fitted on train and evaluated on val

- train: **835** tiles (`/home/priyanshu/outputs/valsplit/A`)
- val: **834** tiles (`/home/priyanshu/outputs/valsplit/B`)
- classes: **7**  |  published τ: **0.5**  |  no tile id in both

SegEarth-OV3 tunes τ per dataset using labels, so fitting thresholds on a train split and evaluating on val is **the same protocol with more parameters**. No model weights are trained; the pipeline stays training-free.

| rule | params | fitted on | train mIoU | **val mIoU** | Δ val |
|---|---|---|---|---|---|
| published τ = 0.5 | 1 | (theirs) | 47.14 | **47.61** | — |
| global τ = 0.425 | 1 | train | 47.30 | **47.48** | -0.13 |
| **per-class τ** | 6 | **train** | 48.53 | **49.05** | **+1.44** |
| _per-class τ (oracle)_ | 6 | _val_ | — | _49.19_ | _+1.59_ |

Generalisation gap on the fitted row: train 48.53 → val 49.05 (**+0.52**). The oracle row is fitted on val and is a ceiling, never a result.

## Thresholds

| class | fitted on train | oracle (val) | published |
|---|---|---|---|
| background | — *(no effect)* | — | — |
| building | **0.325** | 0.430 | 0.5 |
| road | **0.540** | 0.580 | 0.5 |
| water | **0.195** | 0.170 | 0.5 |
| barren | **0.375** | 0.370 | 0.5 |
| forest | **0.445** | 0.440 | 0.5 |
| agricultural | **0.600** | 0.595 | 0.5 |

## Per-class IoU on val, published τ vs fitted

| class | published | fitted | Δ |
|---|---|---|---|
| background | 44.63 | 46.68 | **+2.05** |
| building | 63.28 | 63.62 | **+0.34** |
| road | 54.97 | 55.25 | **+0.28** |
| water | 50.69 | 56.76 | **+6.07** |
| barren | 37.38 | 38.61 | **+1.22** |
| forest | 35.48 | 35.84 | **+0.36** |
| agricultural | 46.83 | 46.61 | **-0.22** |

`background` **+2.05**, the 6 real classes **+8.06** in aggregate.

## Verdict

✅ **+1.44 mIoU on val, with the real classes gaining +8.06 in aggregate** — 91% of the oracle bound, from thresholds fitted on a disjoint train split. Same protocol the baseline uses for its own τ, more parameters, no weights trained.

Before claiming it: re-run on the second dataset, report the generalisation gap above, and give the per-class thresholds so a reader can see the spread a single τ was forced to average over.