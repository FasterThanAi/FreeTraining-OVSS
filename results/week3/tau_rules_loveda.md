# Label-free per-class thresholds

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tiles: **1669**  |  classes: **7**
- published τ: **0.5** → **47.37** mIoU

SegEarth-OV3 tunes one τ per dataset with labels, so a rule that spends **one** label-tuned knob and distributes it across classes by a fixed principle is at parity with the baseline. The oracle row spends **N** independent parameters and is a ceiling, not a competitor.

| rule | knobs | **mIoU** | Δ | share of oracle headroom |
|---|---|---|---|---|
| published τ = 0.5 | 1 (baseline) | **47.37** | — | — |
| best global τ = 0.430 | 1 | **47.41** | +0.04 | 3% |
| per-class Otsu | **0** | **47.19** | -0.17 | -12% |
| equal-commitment, q = 0.36 | 1 | **44.39** | -2.98 | **-204%** |
| presence-scaled, level = 0.475 | 1 | **46.63** | -0.74 | **-50%** |
| **ORACLE per-class** | 6 | **48.83** | **+1.46** | 100% |

## Thresholds chosen — per-class Otsu vs the oracle

| class | label-free τ | oracle τ | IoU published | label-free | Δ |
|---|---|---|---|---|---|
| background | — | — | 45.51 | 45.78 | **+0.28** |
| building | 0.520 | 0.320 | 63.80 | 63.57 | **-0.23** |
| road | 0.515 | 0.575 | 53.88 | 53.93 | **+0.05** |
| water | 0.550 | 0.170 | 51.41 | 49.75 | **-1.66** |
| barren | 0.420 | 0.375 | 35.75 | 36.31 | **+0.55** |
| forest | 0.520 | 0.440 | 33.75 | 33.49 | **-0.25** |
| agricultural | 0.515 | 0.595 | 47.48 | 47.53 | **+0.04** |

`background` **+0.28**, the 6 real classes **-1.50** in aggregate.

## Verdict

⛔ **No label-free rule reaches the bound.** Best is `per-class Otsu` at -0.17 mIoU (-12% of the oracle's +1.46). Per-class thresholds help only when the thresholds are chosen with labels, so this is a property of the evaluation rather than a method. Report the oracle bound as a limitation of global thresholding and stop here.