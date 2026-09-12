# Oracle bound on threshold tuning

- cache: `/home/priyanshu/outputs/oem_tau0.1/cache`  |  tiles: **384**  |  classes: **9**  |  bins: 200
- published τ: **0.1**

## Cross-check

The published-τ row is computed here from the confidence histogram, by a different code path than `measure_discard_rate.py` and `selective_recovery_miou.py`. It must agree with them: **44.16** against 47.37 on the full LoveDA cache, 44.16 on OpenEarthMap. If it does not, the histogram or the label convention is wrong and every row below is void.

⚠️ **The two swept rows are ORACLE BOUNDS.** They choose thresholds using the evaluation labels, so they bound what threshold tuning could ever achieve. They are not methods and must not be quoted as results.

| rung | free parameters | **mIoU** | Δ vs published |
|---|---|---|---|
| published τ = 0.1 | 0 | **44.16** | — |
| best global τ = 0.025 | 1 | **49.31** | +5.15 |
| best per-class τ | 8 | **49.44** | **+5.28** |

## Chosen per-class thresholds

| class | τ | IoU before | after | Δ |
|---|---|---|---|---|
| background | —  *(no effect)* | 17.13 | 66.35 | **+49.22** |
| bareland | 0.010 | 13.77 | 12.95 | **-0.82** |
| grass | 0.030 | 42.92 | 43.27 | **+0.35** |
| pavement | 0.020 | 27.88 | 30.59 | **+2.71** |
| road | 0.025 | 45.88 | 43.73 | **-2.15** |
| tree | 0.025 | 63.91 | 63.69 | **-0.21** |
| water | 0.030 | 66.57 | 66.37 | **-0.20** |
| cropland | 0.055 | 44.08 | 43.91 | **-0.18** |
| building | 0.025 | 75.32 | 74.11 | **-1.21** |

`background` **+49.22**, the 8 real classes **-1.71** in aggregate.

> ⚠️ **The gain is mostly the background row again.** Same pattern as the recovery experiments: mIoU rises because one over-predicted class is corrected, not because land cover is classified better. Report the per-class column beside the headline.

## Verdict

⛔ **Threshold tuning is worth +5.28 mIoU at the oracle bound** — more than the recovery machinery achieves. That reframes the problem: a substantial share of the residual is a CALIBRATION failure, a single global τ being wrong for individual classes in opposite directions, rather than a recovery failure. Before anything else, find out how much of this bound a label-free rule can reach — per-class τ set from the presence score, or from each class's own confidence distribution, needs no ground truth and is worth testing immediately.