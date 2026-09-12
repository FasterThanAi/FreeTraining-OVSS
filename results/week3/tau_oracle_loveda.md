# Oracle bound on threshold tuning

- cache: `/home/priyanshu/outputs/week3_fused/cache`  |  tiles: **1669**  |  classes: **7**  |  bins: 200
- published τ: **0.5**

## Cross-check

The published-τ row is computed here from the confidence histogram, by a different code path than `measure_discard_rate.py` and `selective_recovery_miou.py`. It must agree with them: **47.37** against 47.37 on the full LoveDA cache, 44.16 on OpenEarthMap. If it does not, the histogram or the label convention is wrong and every row below is void.

⚠️ **The two swept rows are ORACLE BOUNDS.** They choose thresholds using the evaluation labels, so they bound what threshold tuning could ever achieve. They are not methods and must not be quoted as results.

| rung | free parameters | **mIoU** | Δ vs published |
|---|---|---|---|
| published τ = 0.5 | 0 | **47.37** | — |
| best global τ = 0.430 | 1 | **47.41** | +0.04 |
| best per-class τ | 6 | **48.83** | **+1.46** |

## Chosen per-class thresholds

| class | τ | IoU before | after | Δ |
|---|---|---|---|---|
| background | —  *(no effect)* | 45.51 | 47.10 | **+1.59** |
| building | 0.320 | 63.80 | 64.57 | **+0.76** |
| road | 0.575 | 53.88 | 53.99 | **+0.11** |
| water | 0.170 | 51.41 | 58.11 | **+6.70** |
| barren | 0.375 | 35.75 | 36.77 | **+1.02** |
| forest | 0.440 | 33.75 | 34.18 | **+0.43** |
| agricultural | 0.595 | 47.48 | 47.10 | **-0.39** |

`background` **+1.59**, the 6 real classes **+8.63** in aggregate.

## Verdict

⚠️ **Threshold tuning is worth +1.46 mIoU at the oracle bound.** Real but modest, and unreachable without labels. Report it as the ceiling on threshold tuning and note that a practical per-class rule would capture only part of it.