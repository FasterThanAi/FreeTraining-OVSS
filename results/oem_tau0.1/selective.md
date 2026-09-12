# Week 3 — does selective recovery move mIoU?

- tiles: **384**  |  τ: **0.1**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **14,063,988**
- atoms: **`slic`**  (oracle ceiling 92.8%)
- region scope: **`all`**  (every background-assigned pixel, as at inference)

## Validation gate

| | this run | expected |
|---|---|---|
| mIoU, recover nothing | **44.16** | 47.37 |
| background-assigned px | **14,063,988** | 323,084,415 |

_(Partial cache — the 47.37 / 323,084,415 gate applies only to the full 1669-tile LoveDA val cache. Not checked here.)_

## Operating points

Abstain unless the region clears both thresholds. `margin` = top1−top2 of the score; `purity` = share of boundary held by the top neighbour class.

Sorted by mIoU. `max px` is the atom size ceiling; `classes` is which labels we are willing to commit to.

| classes | max px | margin ≥ | purity ≥ | recovered px | precision | **mIoU** | Δ |
|---|---|---|---|---|---|---|---|
| all classes | ∞ | 0.0 | 0.0 | 13,831,697 (98.3%) | 27.5% | **46.45** | +2.28 |
| all classes | 10000 | 0.0 | 0.0 | 13,821,364 (98.3%) | 27.5% | **46.44** | +2.27 |
| all classes | 2000 | 0.0 | 0.0 | 11,557,449 (82.2%) | 29.3% | **46.35** | +2.18 |
| all classes | 500 | 0.0 | 0.0 | 4,713,399 (33.5%) | 31.3% | **44.80** | +0.63 |
| all classes | 2000 | 1.0 | 0.0 | 5,802,296 (41.3%) | 32.8% | **44.64** | +0.47 |
| all classes | 500 | 1.0 | 0.0 | 2,111,214 (15.0%) | 35.4% | **44.41** | +0.25 |
| all classes | 2000 | 0.0 | 0.7 | 2,787,961 (19.8%) | 40.2% | **44.32** | +0.16 |
| all classes | 2000 | 1.0 | 0.7 | 2,787,961 (19.8%) | 40.2% | **44.32** | +0.16 |
| all classes | 10000 | 1.0 | 0.0 | 7,223,645 (51.4%) | 30.0% | **44.29** | +0.12 |
| all classes | ∞ | 1.0 | 0.0 | 7,223,645 (51.4%) | 30.0% | **44.29** | +0.12 |
| all classes | 500 | 0.0 | 0.7 | 875,568 (6.2%) | 43.5% | **44.28** | +0.12 |
| all classes | 500 | 1.0 | 0.7 | 875,568 (6.2%) | 43.5% | **44.28** | +0.12 |
| all classes | 10000 | 0.0 | 0.7 | 3,708,028 (26.4%) | 34.9% | **44.01** | -0.15 |
| all classes | ∞ | 0.0 | 0.7 | 3,708,028 (26.4%) | 34.9% | **44.01** | -0.15 |
| all classes | 10000 | 1.0 | 0.7 | 3,708,028 (26.4%) | 34.9% | **44.01** | -0.15 |
| all classes | ∞ | 1.0 | 0.7 | 3,708,028 (26.4%) | 34.9% | **44.01** | -0.15 |

## Per-class IoU at the best operating point

| class | before | after | Δ |
|---|---|---|---|
| background | 17.13 | 39.80 | **+22.67** |
| bareland | 13.77 | 13.40 | **-0.36** |
| grass | 42.92 | 43.15 | **+0.23** |
| pavement | 27.88 | 29.94 | **+2.05** |
| road | 45.88 | 45.32 | **-0.56** |
| tree | 63.91 | 63.26 | **-0.64** |
| water | 66.57 | 67.60 | **+1.02** |
| cropland | 44.08 | 43.98 | **-0.10** |
| building | 75.32 | 71.57 | **-3.75** |

`background` accounts for **110%** of the total IoU gain. Real classes that improved by >0.1: **grass, pavement, water**.

> ⚠️ **Nearly all of the gain is the background row.** The method is unwinding an over-prediction rather than recovering land cover. That is a real mIoU gain and a much weaker claim than it looks; report the per-class table beside the headline.


## Verdict

✅ **Best: classes `all classes`, max atom ∞ px, margin ≥ 0.0, purity ≥ 0.0 → mIoU 46.45 (+2.28)**, recovering 13,831,697 pixels (98.3% of the residual) at **27.5% precision**.

Compare against the alternatives already measured on LoveDA: τ→0.1 gives −5.54 at 36.6% precision, presence removal −11.97.

⚠️ **Note the best row abstains from nothing** (margin 0, purity 0, no size ceiling). On this dataset selectivity is not the contribution — recovering everything reachable wins, and any abstention only costs. Do not carry the "calibrated abstention" framing here.

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.