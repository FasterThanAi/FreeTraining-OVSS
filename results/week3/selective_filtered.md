# Week 3 — does selective recovery move mIoU?

- tiles: **1669**  |  τ: **0.5**  |  β: **0.25**  |  min component: **64px**
- background-assigned real-class pixels: **323,084,415**
- atoms: **`slic`**  (oracle ceiling 92.8%)
- region scope: **`all`**  (every background-assigned pixel, as at inference)

## Validation gate

| | this run | expected |
|---|---|---|
| mIoU, recover nothing | **47.37** | 47.37 |
| background-assigned px | **323,084,415** | 323,084,415 |

✅ **Gate passed** — this script's confusion bookkeeping agrees with `measure_discard_rate.py`, so the rows below are trustworthy.

## Operating points

Abstain unless the region clears both thresholds. `margin` = top1−top2 of the score; `purity` = share of boundary held by the top neighbour class.

Sorted by mIoU. `max px` is the atom size ceiling; `classes` is which labels we are willing to commit to.

| classes | max px | margin ≥ | purity ≥ | recovered px | precision | **mIoU** | Δ |
|---|---|---|---|---|---|---|---|
| all classes | 500 | 0.0 | 0.7 | 18,577,350 (5.7%) | 40.0% | **47.41** | +0.04 |
| all classes | 500 | 1.0 | 0.7 | 18,577,350 (5.7%) | 40.0% | **47.41** | +0.04 |
| bld+wat+road+barren | 500 | 0.0 | 0.7 | 9,452,016 (2.9%) | 40.2% | **47.40** | +0.03 |
| bld+wat+road+barren | 500 | 1.0 | 0.7 | 9,452,016 (2.9%) | 40.2% | **47.40** | +0.03 |
| bld+wat+road+barren | 500 | 1.0 | 0.0 | 10,060,304 (3.1%) | 39.5% | **47.39** | +0.02 |
| all classes | 500 | 1.0 | 0.0 | 19,720,987 (6.1%) | 39.3% | **47.39** | +0.02 |
| bld+wat+road | 500 | 0.0 | 0.7 | 7,804,642 (2.4%) | 42.1% | **47.39** | +0.02 |
| bld+wat+road | 500 | 1.0 | 0.7 | 7,804,642 (2.4%) | 42.1% | **47.39** | +0.02 |
| bld+wat+road | 500 | 1.0 | 0.0 | 8,222,606 (2.5%) | 41.6% | **47.38** | +0.01 |
| building+water | 500 | 1.0 | 0.0 | 5,858,954 (1.8%) | 43.5% | **47.38** | +0.01 |
| building+water | 500 | 0.0 | 0.7 | 5,619,216 (1.7%) | 43.5% | **47.38** | +0.01 |
| building+water | 500 | 1.0 | 0.7 | 5,619,216 (1.7%) | 43.5% | **47.38** | +0.01 |
| building+water | 500 | 0.0 | 0.0 | 7,048,065 (2.2%) | 42.9% | **47.38** | +0.01 |
| bld+wat+road+barren | 500 | 0.0 | 0.0 | 13,037,237 (4.0%) | 37.3% | **47.36** | -0.01 |
| bld+wat+road | 500 | 0.0 | 0.0 | 10,497,524 (3.2%) | 39.4% | **47.35** | -0.02 |
| building | 500 | 0.0 | 0.7 | 2,924,533 (0.9%) | 36.6% | **47.34** | -0.03 |
| building | 500 | 1.0 | 0.7 | 2,924,533 (0.9%) | 36.6% | **47.34** | -0.03 |
| building | 500 | 1.0 | 0.0 | 3,008,691 (0.9%) | 36.3% | **47.34** | -0.03 |
| all classes | 500 | 0.0 | 0.0 | 25,086,664 (7.8%) | 36.9% | **47.34** | -0.03 |
| building | 500 | 0.0 | 0.0 | 3,587,916 (1.1%) | 34.7% | **47.32** | -0.05 |

## Verdict

✅ **Best: classes `all classes`, max atom 500 px, margin ≥ 0.0, purity ≥ 0.7 → mIoU 47.41 (+0.04)**, recovering 18,577,350 pixels (5.7% of the residual) at **40.0% precision**.

Compare against the alternatives already measured: τ→0.1 gives −5.54 at 36.6% precision, presence removal −11.97. **Selective recovery is the first intervention that reaches the residual and does not cost more than it returns.** The abstention rule is the contribution — threshold relaxation and DenseCRF both commit everywhere and that is exactly why they lose.

> Report recovery rate and precision separately (WEEK1_RESULTS §12). A headline mIoU gain without the precision column invites the objection the τ-sweep already answers.