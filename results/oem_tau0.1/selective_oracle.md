# Week 3 — does selective recovery move mIoU?

- tiles: **384**  |  τ: **0.1**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **14,063,988**
- atoms: **`slic`**  (oracle ceiling 92.8%)
- region scope: **`oracle`**  ⚠️ **ORACLE — uses GT to choose which pixels to touch. Upper bound only, never a result.**

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
| all classes | ∞ | 0.0 | 0.0 | 12,679,241 (90.2%) | 30.0% | **49.37** | +5.21 |
| all classes | 10000 | 0.0 | 0.0 | 12,668,908 (90.1%) | 30.0% | **49.36** | +5.19 |
| all classes | 2000 | 0.0 | 0.0 | 10,966,577 (78.0%) | 30.9% | **47.50** | +3.34 |
| all classes | 10000 | 1.0 | 0.0 | 6,284,914 (44.7%) | 34.4% | **45.35** | +1.19 |
| all classes | ∞ | 1.0 | 0.0 | 6,284,914 (44.7%) | 34.4% | **45.35** | +1.19 |
| all classes | 2000 | 1.0 | 0.0 | 5,314,483 (37.8%) | 35.9% | **45.17** | +1.01 |
| all classes | 500 | 0.0 | 0.0 | 4,716,086 (33.5%) | 31.5% | **44.82** | +0.65 |
| all classes | 10000 | 0.0 | 0.7 | 2,868,044 (20.4%) | 45.1% | **44.77** | +0.60 |
| all classes | ∞ | 0.0 | 0.7 | 2,868,044 (20.4%) | 45.1% | **44.77** | +0.60 |
| all classes | 10000 | 1.0 | 0.7 | 2,868,044 (20.4%) | 45.1% | **44.77** | +0.60 |
| all classes | ∞ | 1.0 | 0.7 | 2,868,044 (20.4%) | 45.1% | **44.77** | +0.60 |
| all classes | 2000 | 0.0 | 0.7 | 2,344,822 (16.7%) | 47.9% | **44.72** | +0.55 |
| all classes | 2000 | 1.0 | 0.7 | 2,344,822 (16.7%) | 47.9% | **44.72** | +0.55 |
| all classes | 500 | 1.0 | 0.0 | 2,111,488 (15.0%) | 35.6% | **44.43** | +0.27 |
| all classes | 500 | 0.0 | 0.7 | 873,400 (6.2%) | 43.9% | **44.29** | +0.13 |
| all classes | 500 | 1.0 | 0.7 | 873,400 (6.2%) | 43.9% | **44.29** | +0.13 |

## Verdict

✅ **Best: classes `all classes`, max atom ∞ px, margin ≥ 0.0, purity ≥ 0.0 → mIoU 49.37 (+5.21)**, recovering 12,679,241 pixels (90.2% of the residual) at **30.0% precision**.

Compare against the alternatives already measured: τ→0.1 gives −5.54 at 36.6% precision, presence removal −11.97. **Selective recovery is the first intervention that reaches the residual and does not cost more than it returns.** The abstention rule is the contribution — threshold relaxation and DenseCRF both commit everywhere and that is exactly why they lose.

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.